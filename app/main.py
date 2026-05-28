from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

TF_IMPORT_ERROR = None
TF_KERAS_LOAD_ERROR = None
TF_LOAD_MODEL = None

try:
    import tensorflow as tf
except Exception as exc:
    tf = None
    TF_IMPORT_ERROR = str(exc)

if tf is not None:
    try:
        from tensorflow.keras.models import load_model as _tf_load_model
        TF_LOAD_MODEL = _tf_load_model
    except Exception as exc:
        TF_KERAS_LOAD_ERROR = str(exc)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / "model"

CNN_MODEL_PATH = MODEL_DIR / "modelo_cnn_actividad.keras"
RNN_MODEL_PATH = MODEL_DIR / "modelo_rnn_caidas.keras"

FEATURE_COLUMNS = ("acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z")
ACTIVITY_CLASSES = (
    "bajando_escaleras",
    "caminando",
    "parado",
    "sentado",
    "subiendo_escaleras",
    "trotando",
)

CNN_WINDOW_SECONDS = float(os.getenv("CNN_WINDOW_SECONDS", "2.0"))
RNN_WINDOW_SECONDS = float(os.getenv("RNN_WINDOW_SECONDS", "20.0"))
FALL_PEAK_THRESHOLD = float(os.getenv("FALL_PEAK_THRESHOLD", "3.0"))
FALL_PROBABILITY_THRESHOLD = float(os.getenv("FALL_PROBABILITY_THRESHOLD", "0.5"))


class SensorSample(BaseModel):
    ts_ms: float | None = None
    acc_x: float
    acc_y: float
    acc_z: float
    gyr_x: float
    gyr_y: float
    gyr_z: float


class InferenceRequest(BaseModel):
    samples: list[SensorSample] = Field(default_factory=list)
    sample_rate_hz: float = 50.0


class ModelRegistry:
    def __init__(self) -> None:
        self.cnn_model: Any | None = None
        self.rnn_model: Any | None = None
        self.ready: bool = False
        self.error: str | None = None
        self.keras_backend_name: str = "tensorflow.keras"
        self.cnn_steps: int = 128
        self.rnn_steps: int = 128
        self.features: int = len(FEATURE_COLUMNS)
        self.lock = threading.Lock()

    def load(self) -> None:
        if tf is None:
            self.ready = False
            self.error = (
                "TensorFlow no disponible. "
                f"TF_ERROR={TF_IMPORT_ERROR}"
            )
            return
        if TF_LOAD_MODEL is None:
            self.ready = False
            self.error = (
                "No se pudo cargar tensorflow.keras.models.load_model. "
                f"TF_KERAS_ERROR={TF_KERAS_LOAD_ERROR}"
            )
            return

        if not CNN_MODEL_PATH.exists() or not RNN_MODEL_PATH.exists():
            self.ready = False
            self.error = (
                "No se encontraron modelos .keras. "
                f"CNN: {CNN_MODEL_PATH.exists()} | RNN: {RNN_MODEL_PATH.exists()}"
            )
            return

        try:
            self.cnn_model = TF_LOAD_MODEL(CNN_MODEL_PATH)
            self.rnn_model = TF_LOAD_MODEL(RNN_MODEL_PATH)

            cnn_shape = self.cnn_model.input_shape
            rnn_shape = self.rnn_model.input_shape
            self.cnn_steps = int(cnn_shape[1])
            self.rnn_steps = int(rnn_shape[1])
            self.features = int(cnn_shape[2])

            if self.features != len(FEATURE_COLUMNS):
                self.ready = False
                self.error = (
                    "El modelo espera un numero de features distinto al frontend. "
                    f"Modelo={self.features}, Front={len(FEATURE_COLUMNS)}"
                )
                return

            self.ready = True
            self.error = None
        except Exception as exc:
            self.ready = False
            self.error = f"Error cargando modelos: {exc}"


MODEL_REGISTRY = ModelRegistry()
MODEL_REGISTRY.load()

app = FastAPI(title="Vitalia Motion Sensor")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def _samples_to_arrays(samples: list[SensorSample]) -> tuple[np.ndarray, np.ndarray]:
    rows: list[list[float]] = []
    ts_list: list[float] = []
    for idx, s in enumerate(samples):
        rows.append([s.acc_x, s.acc_y, s.acc_z, s.gyr_x, s.gyr_y, s.gyr_z])
        ts_list.append(float(s.ts_ms if s.ts_ms is not None else idx))
    return np.asarray(rows, dtype=np.float32), np.asarray(ts_list, dtype=np.float64)


def _slice_last_seconds(values: np.ndarray, ts_ms: np.ndarray, seconds: float, fallback_hz: float) -> np.ndarray:
    if len(values) == 0:
        return values

    end_ts = float(ts_ms[-1])
    start_ts = end_ts - seconds * 1000.0
    mask = ts_ms >= start_ts
    out = values[mask]

    if len(out) == 0:
        n = max(1, int(round(seconds * max(fallback_hz, 1.0))))
        out = values[-n:]

    return out


def _resample_to_steps(values: np.ndarray, target_steps: int) -> np.ndarray:
    if len(values) == 0:
        return np.zeros((target_steps, values.shape[1]), dtype=np.float32)
    if len(values) == target_steps:
        return values.astype(np.float32)
    if len(values) == 1:
        return np.repeat(values, target_steps, axis=0).astype(np.float32)

    src_idx = np.linspace(0.0, 1.0, num=len(values), dtype=np.float64)
    dst_idx = np.linspace(0.0, 1.0, num=target_steps, dtype=np.float64)
    out = np.zeros((target_steps, values.shape[1]), dtype=np.float32)
    for col in range(values.shape[1]):
        out[:, col] = np.interp(dst_idx, src_idx, values[:, col]).astype(np.float32)
    return out


def _zscore_per_window(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (values - mean) / std


def _acc_peak(window: np.ndarray) -> float:
    mag = np.sqrt(np.sum(np.square(window[:, :3]), axis=1))
    return float(np.max(mag))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/model-status")
async def model_status() -> dict[str, Any]:
    return {
        "ready": MODEL_REGISTRY.ready,
        "error": MODEL_REGISTRY.error,
        "keras_backend": MODEL_REGISTRY.keras_backend_name,
        "tensorflow_module_path": getattr(tf, "__file__", None) if tf is not None else None,
        "tensorflow_version": getattr(tf, "__version__", None) if tf is not None else None,
        "tensorflow_has_keras": bool(hasattr(tf, "keras")) if tf is not None else False,
        "tensorflow_import_error": TF_IMPORT_ERROR,
        "tensorflow_keras_load_error": TF_KERAS_LOAD_ERROR,
        "cnn_model_path": str(CNN_MODEL_PATH),
        "rnn_model_path": str(RNN_MODEL_PATH),
        "cnn_steps": MODEL_REGISTRY.cnn_steps,
        "rnn_steps": MODEL_REGISTRY.rnn_steps,
        "features": MODEL_REGISTRY.features,
        "activity_classes": list(ACTIVITY_CLASSES),
    }


@app.post("/api/infer")
async def infer(payload: InferenceRequest) -> dict[str, Any]:
    if not MODEL_REGISTRY.ready or MODEL_REGISTRY.cnn_model is None or MODEL_REGISTRY.rnn_model is None:
        return {
            "ready": False,
            "error": MODEL_REGISTRY.error or "Modelos no disponibles.",
            "state": "modelo_no_disponible",
        }

    if len(payload.samples) < 8:
        return {"ready": True, "state": "esperando_mas_datos", "detail": "Muestras insuficientes para inferencia."}

    values, ts_ms = _samples_to_arrays(payload.samples)

    short_raw = _slice_last_seconds(values, ts_ms, CNN_WINDOW_SECONDS, payload.sample_rate_hz)
    long_raw = _slice_last_seconds(values, ts_ms, RNN_WINDOW_SECONDS, payload.sample_rate_hz)

    short_rs = _resample_to_steps(short_raw, MODEL_REGISTRY.cnn_steps)
    long_rs = _resample_to_steps(long_raw, MODEL_REGISTRY.rnn_steps)

    short_input = _zscore_per_window(short_rs)[None, :, :]
    long_input = _zscore_per_window(long_rs)[None, :, :]

    peak = _acc_peak(short_raw if len(short_raw) > 0 else short_rs)
    possible_fall = peak >= FALL_PEAK_THRESHOLD

    with MODEL_REGISTRY.lock:
        cnn_probs = MODEL_REGISTRY.cnn_model.predict(short_input, verbose=0)[0]
        activity_idx = int(np.argmax(cnn_probs))
        activity_label = ACTIVITY_CLASSES[activity_idx] if activity_idx < len(ACTIVITY_CLASSES) else str(activity_idx)
        activity_confidence = float(cnn_probs[activity_idx])

        fall_probability = None
        fall_confirmed = False
        rnn_evaluated = False
        if possible_fall:
            fall_probability = float(MODEL_REGISTRY.rnn_model.predict(long_input, verbose=0)[0, 0])
            fall_confirmed = fall_probability >= FALL_PROBABILITY_THRESHOLD
            rnn_evaluated = True

    if fall_confirmed:
        state = "caida_detectada"
    elif possible_fall:
        state = "posible_caida_en_revision"
    else:
        state = activity_label

    return {
        "ready": True,
        "state": state,
        "activity_label": activity_label,
        "activity_confidence": activity_confidence,
        "activity_probabilities": {label: float(cnn_probs[i]) for i, label in enumerate(ACTIVITY_CLASSES)},
        "fall_peak_triggered": possible_fall,
        "fall_peak_value": peak,
        "rnn_evaluated": rnn_evaluated,
        "fall_probability": fall_probability,
        "fall_confirmed": fall_confirmed,
    }
