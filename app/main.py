from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

import joblib
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
CNN_SCALER_PATH = MODEL_DIR / "scaler_cnn.joblib"
CNN_CLASSES_PATH = MODEL_DIR / "activity_classes.json"
CNN_METADATA_PATH = MODEL_DIR / "cnn_metadata.json"

RNN_MODEL_PATH = MODEL_DIR / "modelo_rnn_caidas.keras"
RNN_SCALER_PATH = MODEL_DIR / "scaler_rnn.joblib"
RNN_METADATA_PATH = MODEL_DIR / "rnn_metadata.json"
RNN_THRESHOLD_REPORT_PATH = MODEL_DIR / "rnn_threshold_report.json"

ALT_CNN_MODEL_PATH = MODEL_DIR / "model" / "modelo_cnn_actividad.keras"
ALT_CNN_SCALER_PATH = MODEL_DIR / "model" / "scaler_cnn.joblib"
ALT_CNN_CLASSES_PATH = MODEL_DIR / "model" / "activity_classes.json"
ALT_CNN_METADATA_PATH = MODEL_DIR / "model" / "cnn_metadata.json"

ALT_RNN_MODEL_PATH = MODEL_DIR / "model" / "modelo_rnn_caidas.keras"
ALT_RNN_SCALER_PATH = MODEL_DIR / "model" / "scaler_rnn.joblib"
ALT_RNN_METADATA_PATH = MODEL_DIR / "model" / "rnn_metadata.json"
ALT_RNN_THRESHOLD_REPORT_PATH = MODEL_DIR / "model" / "rnn_threshold_report.json"

RAW_FEATURE_COLUMNS = ("acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z")
MODEL_FEATURE_COLUMNS = ("acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z", "acc_mag", "gyr_mag")
ACTIVITY_CLASSES_FALLBACK = (
    "bajando_escaleras",
    "caminando",
    "parado",
    "sentado",
    "subiendo_escaleras",
    "trotando",
)

DEFAULT_CNN_WINDOW_SECONDS = float(os.getenv("CNN_WINDOW_SECONDS", "2.0"))
DEFAULT_RNN_WINDOW_SECONDS = float(os.getenv("RNN_WINDOW_SECONDS", "30.0"))
FALL_PEAK_THRESHOLD = float(os.getenv("FALL_PEAK_THRESHOLD", "3.0"))
DEFAULT_FALL_PROBABILITY_THRESHOLD = float(os.getenv("FALL_PROBABILITY_THRESHOLD", "0.6"))
FALL_PEAK_DELTA_THRESHOLD = float(os.getenv("FALL_PEAK_DELTA_THRESHOLD", "1.2"))
FALL_CONFIRM_MARGIN = float(os.getenv("FALL_CONFIRM_MARGIN", "0.1"))
ACC_MS2_TO_G = 1.0 / 9.80665
DEG_TO_RAD = np.pi / 180.0


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


def _resolve_existing_path(primary: Path, alternative: Path, prefer_alternative: bool = True) -> Path:
    first, second = (alternative, primary) if prefer_alternative else (primary, alternative)
    if first.exists():
        return first
    return second


class ModelRegistry:
    def __init__(self) -> None:
        self.cnn_model: Any | None = None
        self.rnn_model: Any | None = None
        self.cnn_scaler: Any | None = None
        self.rnn_scaler: Any | None = None

        self.activity_classes: list[str] = list(ACTIVITY_CLASSES_FALLBACK)
        self.cnn_metadata: dict[str, Any] = {}
        self.rnn_metadata: dict[str, Any] = {}
        self.rnn_threshold_report: dict[str, Any] = {}

        self.ready: bool = False
        self.error: str | None = None
        self.keras_backend_name: str = "tensorflow.keras"

        self.cnn_steps: int = 128
        self.rnn_steps: int = 128
        self.cnn_features: int = len(MODEL_FEATURE_COLUMNS)
        self.rnn_features: int = len(MODEL_FEATURE_COLUMNS)
        self.cnn_window_seconds: float = DEFAULT_CNN_WINDOW_SECONDS
        self.rnn_window_seconds: float = DEFAULT_RNN_WINDOW_SECONDS
        self.fall_probability_threshold: float = DEFAULT_FALL_PROBABILITY_THRESHOLD

        self.cnn_model_path: Path = CNN_MODEL_PATH
        self.cnn_scaler_path: Path = CNN_SCALER_PATH
        self.cnn_classes_path: Path = CNN_CLASSES_PATH
        self.cnn_metadata_path: Path = CNN_METADATA_PATH
        self.rnn_model_path: Path = RNN_MODEL_PATH
        self.rnn_scaler_path: Path = RNN_SCALER_PATH
        self.rnn_metadata_path: Path = RNN_METADATA_PATH
        self.rnn_threshold_report_path: Path = RNN_THRESHOLD_REPORT_PATH

        self.lock = threading.Lock()

    def load(self) -> None:
        if tf is None:
            self.ready = False
            self.error = f"TensorFlow no disponible. TF_ERROR={TF_IMPORT_ERROR}"
            return

        if TF_LOAD_MODEL is None:
            self.ready = False
            self.error = (
                "No se pudo cargar tensorflow.keras.models.load_model. "
                f"TF_KERAS_ERROR={TF_KERAS_LOAD_ERROR}"
            )
            return

        self.cnn_model_path = _resolve_existing_path(CNN_MODEL_PATH, ALT_CNN_MODEL_PATH)
        self.cnn_scaler_path = _resolve_existing_path(CNN_SCALER_PATH, ALT_CNN_SCALER_PATH)
        self.cnn_classes_path = _resolve_existing_path(CNN_CLASSES_PATH, ALT_CNN_CLASSES_PATH)
        self.cnn_metadata_path = _resolve_existing_path(CNN_METADATA_PATH, ALT_CNN_METADATA_PATH)
        self.rnn_model_path = _resolve_existing_path(RNN_MODEL_PATH, ALT_RNN_MODEL_PATH)
        self.rnn_scaler_path = _resolve_existing_path(RNN_SCALER_PATH, ALT_RNN_SCALER_PATH)
        self.rnn_metadata_path = _resolve_existing_path(RNN_METADATA_PATH, ALT_RNN_METADATA_PATH)
        self.rnn_threshold_report_path = _resolve_existing_path(
            RNN_THRESHOLD_REPORT_PATH, ALT_RNN_THRESHOLD_REPORT_PATH
        )

        required_paths = {
            "cnn_model": self.cnn_model_path,
            "cnn_scaler": self.cnn_scaler_path,
            "cnn_classes": self.cnn_classes_path,
            "cnn_metadata": self.cnn_metadata_path,
            "rnn_model": self.rnn_model_path,
            "rnn_scaler": self.rnn_scaler_path,
            "rnn_metadata": self.rnn_metadata_path,
            "rnn_threshold_report": self.rnn_threshold_report_path,
        }
        missing = [name for name, path in required_paths.items() if not path.exists()]
        if missing:
            self.ready = False
            self.error = f"Faltan archivos requeridos: {', '.join(missing)}"
            return

        try:
            self.cnn_model = TF_LOAD_MODEL(self.cnn_model_path)
            self.rnn_model = TF_LOAD_MODEL(self.rnn_model_path)
            self.cnn_scaler = joblib.load(self.cnn_scaler_path)
            self.rnn_scaler = joblib.load(self.rnn_scaler_path)

            with self.cnn_classes_path.open("r", encoding="utf-8") as f:
                classes = json.load(f)
            if isinstance(classes, list) and classes:
                self.activity_classes = [str(x) for x in classes]
            else:
                self.activity_classes = list(ACTIVITY_CLASSES_FALLBACK)

            with self.cnn_metadata_path.open("r", encoding="utf-8") as f:
                self.cnn_metadata = json.load(f)
            if not isinstance(self.cnn_metadata, dict):
                self.cnn_metadata = {}

            with self.rnn_metadata_path.open("r", encoding="utf-8") as f:
                self.rnn_metadata = json.load(f)
            if not isinstance(self.rnn_metadata, dict):
                self.rnn_metadata = {}

            with self.rnn_threshold_report_path.open("r", encoding="utf-8") as f:
                self.rnn_threshold_report = json.load(f)
            if not isinstance(self.rnn_threshold_report, dict):
                self.rnn_threshold_report = {}

            cnn_shape = self.cnn_model.input_shape
            rnn_shape = self.rnn_model.input_shape
            self.cnn_steps = int(cnn_shape[1])
            self.rnn_steps = int(rnn_shape[1])
            self.cnn_features = int(cnn_shape[2])
            self.rnn_features = int(rnn_shape[2])

            if self.cnn_features != len(MODEL_FEATURE_COLUMNS):
                self.ready = False
                self.error = f"CNN espera {self.cnn_features} features, se esperaban {len(MODEL_FEATURE_COLUMNS)}."
                return

            if self.rnn_features != len(MODEL_FEATURE_COLUMNS):
                self.ready = False
                self.error = f"RNN espera {self.rnn_features} features, se esperaban {len(MODEL_FEATURE_COLUMNS)}."
                return

            self.cnn_window_seconds = float(self.cnn_metadata.get("window_seconds", DEFAULT_CNN_WINDOW_SECONDS))
            self.rnn_window_seconds = float(self.rnn_metadata.get("window_seconds", DEFAULT_RNN_WINDOW_SECONDS))
            self.fall_probability_threshold = float(
                self.rnn_metadata.get("recommended_threshold", DEFAULT_FALL_PROBABILITY_THRESHOLD)
            )

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


def _samples_to_model_arrays(samples: list[SensorSample]) -> tuple[np.ndarray, np.ndarray]:
    rows: list[list[float]] = []
    ts_list: list[float] = []
    for idx, s in enumerate(samples):
        acc_mag = float(np.sqrt(s.acc_x**2 + s.acc_y**2 + s.acc_z**2))
        gyr_mag = float(np.sqrt(s.gyr_x**2 + s.gyr_y**2 + s.gyr_z**2))
        rows.append([s.acc_x, s.acc_y, s.acc_z, s.gyr_x, s.gyr_y, s.gyr_z, acc_mag, gyr_mag])
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


def _acc_peak(window: np.ndarray) -> float:
    mag = np.sqrt(np.sum(np.square(window[:, :3]), axis=1))
    return float(np.max(mag))


def _acc_peak_delta(window: np.ndarray) -> float:
    if len(window) == 0:
        return 0.0
    mag = np.sqrt(np.sum(np.square(window[:, :3]), axis=1))
    return float(np.max(mag) - np.median(mag))


def _normalize_units_for_model(values_8: np.ndarray) -> np.ndarray:
    """
    Normaliza unidades de sensores para acercarlas al dominio de entrenamiento:
    - acc: g
    - gyro: rad/s
    Aplica conversion solo si detecta escalas tipicas de m/s^2 o deg/s.
    """
    if len(values_8) == 0:
        return values_8

    out = values_8.astype(np.float32).copy()
    acc = out[:, 0:3]
    gyr = out[:, 3:6]

    acc_norm_med = float(np.median(np.linalg.norm(acc, axis=1)))
    if acc_norm_med > 3.0:
        out[:, 0:3] = out[:, 0:3] * ACC_MS2_TO_G

    gyr_abs_p95 = float(np.percentile(np.abs(gyr), 95))
    if gyr_abs_p95 > 20.0:
        out[:, 3:6] = out[:, 3:6] * DEG_TO_RAD

    out[:, 6] = np.sqrt(np.sum(np.square(out[:, 0:3]), axis=1))
    out[:, 7] = np.sqrt(np.sum(np.square(out[:, 3:6]), axis=1))
    return out


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
        "cnn_model_path": str(MODEL_REGISTRY.cnn_model_path),
        "rnn_model_path": str(MODEL_REGISTRY.rnn_model_path),
        "cnn_scaler_path": str(MODEL_REGISTRY.cnn_scaler_path),
        "rnn_scaler_path": str(MODEL_REGISTRY.rnn_scaler_path),
        "cnn_metadata_path": str(MODEL_REGISTRY.cnn_metadata_path),
        "rnn_metadata_path": str(MODEL_REGISTRY.rnn_metadata_path),
        "rnn_threshold_report_path": str(MODEL_REGISTRY.rnn_threshold_report_path),
        "cnn_model_loaded": MODEL_REGISTRY.cnn_model is not None,
        "rnn_model_loaded": MODEL_REGISTRY.rnn_model is not None,
        "cnn_scaler_loaded": MODEL_REGISTRY.cnn_scaler is not None,
        "rnn_scaler_loaded": MODEL_REGISTRY.rnn_scaler is not None,
        "cnn_metadata_loaded": bool(MODEL_REGISTRY.cnn_metadata),
        "rnn_metadata_loaded": bool(MODEL_REGISTRY.rnn_metadata),
        "cnn_steps": MODEL_REGISTRY.cnn_steps,
        "rnn_steps": MODEL_REGISTRY.rnn_steps,
        "cnn_features": MODEL_REGISTRY.cnn_features,
        "rnn_features": MODEL_REGISTRY.rnn_features,
        "cnn_window_seconds": MODEL_REGISTRY.cnn_window_seconds,
        "rnn_window_seconds": MODEL_REGISTRY.rnn_window_seconds,
        "model_feature_columns": list(MODEL_FEATURE_COLUMNS),
        "raw_feature_columns": list(RAW_FEATURE_COLUMNS),
        "activity_classes": list(MODEL_REGISTRY.activity_classes),
        "fall_probability_threshold": MODEL_REGISTRY.fall_probability_threshold,
        "cnn_metadata": MODEL_REGISTRY.cnn_metadata,
        "rnn_metadata": MODEL_REGISTRY.rnn_metadata,
    }


@app.post("/api/infer")
async def infer(payload: InferenceRequest) -> dict[str, Any]:
    if (
        not MODEL_REGISTRY.ready
        or MODEL_REGISTRY.cnn_model is None
        or MODEL_REGISTRY.rnn_model is None
        or MODEL_REGISTRY.cnn_scaler is None
        or MODEL_REGISTRY.rnn_scaler is None
    ):
        return {
            "ready": False,
            "error": MODEL_REGISTRY.error or "Modelos no disponibles.",
            "state": "modelo_no_disponible",
        }

    if len(payload.samples) < 8:
        return {"ready": True, "state": "esperando_mas_datos", "detail": "Muestras insuficientes para inferencia."}

    values_8, ts_ms = _samples_to_model_arrays(payload.samples)
    values_8 = _normalize_units_for_model(values_8)

    short_raw = _slice_last_seconds(values_8, ts_ms, MODEL_REGISTRY.cnn_window_seconds, payload.sample_rate_hz)
    long_raw = _slice_last_seconds(values_8, ts_ms, MODEL_REGISTRY.rnn_window_seconds, payload.sample_rate_hz)

    short_rs = _resample_to_steps(short_raw, MODEL_REGISTRY.cnn_steps)
    long_rs = _resample_to_steps(long_raw, MODEL_REGISTRY.rnn_steps)

    short_scaled = MODEL_REGISTRY.cnn_scaler.transform(short_rs)
    long_scaled = MODEL_REGISTRY.rnn_scaler.transform(long_rs)

    short_input = short_scaled[None, :, :].astype(np.float32)
    long_input = long_scaled[None, :, :].astype(np.float32)

    short_for_peak = short_raw if len(short_raw) > 0 else short_rs
    peak = _acc_peak(short_for_peak)
    peak_delta = _acc_peak_delta(short_for_peak)
    possible_fall = (peak >= FALL_PEAK_THRESHOLD) and (peak_delta >= FALL_PEAK_DELTA_THRESHOLD)

    with MODEL_REGISTRY.lock:
        cnn_probs = MODEL_REGISTRY.cnn_model.predict(short_input, verbose=0)[0]
        activity_idx = int(np.argmax(cnn_probs))
        if activity_idx < len(MODEL_REGISTRY.activity_classes):
            activity_label = MODEL_REGISTRY.activity_classes[activity_idx]
        else:
            activity_label = str(activity_idx)
        activity_confidence = float(cnn_probs[activity_idx])

        fall_probability = None
        fall_confirmed = False
        rnn_evaluated = False
        warning_fall = False
        if possible_fall:
            fall_probability = float(MODEL_REGISTRY.rnn_model.predict(long_input, verbose=0)[0, 0])
            confirm_threshold = MODEL_REGISTRY.fall_probability_threshold + FALL_CONFIRM_MARGIN
            warning_threshold = max(0.5, MODEL_REGISTRY.fall_probability_threshold * 0.85)
            fall_confirmed = fall_probability >= confirm_threshold
            warning_fall = fall_probability >= warning_threshold
            rnn_evaluated = True

    if fall_confirmed:
        state = "caida_detectada"
    elif possible_fall and warning_fall:
        state = "posible_caida_en_revision"
    else:
        state = activity_label

    return {
        "ready": True,
        "model_type_activity": "residual_cnn_activity",
        "model_type_fall": "rnn_fall_lstm",
        "state": state,
        "activity_label": activity_label,
        "activity_confidence": activity_confidence,
        "activity_probabilities": {
            (MODEL_REGISTRY.activity_classes[i] if i < len(MODEL_REGISTRY.activity_classes) else f"class_{i}"): float(
                cnn_probs[i]
            )
            for i in range(len(cnn_probs))
        },
        "cnn_features_used": int(short_input.shape[2]),
        "cnn_window_steps": int(short_input.shape[1]),
        "rnn_features_used": int(long_input.shape[2]),
        "rnn_window_steps": int(long_input.shape[1]),
        "fall_peak_triggered": possible_fall,
        "fall_peak_value": peak,
        "fall_peak_delta": peak_delta,
        "rnn_evaluated": rnn_evaluated,
        "fall_probability": fall_probability,
        "fall_probability_threshold": MODEL_REGISTRY.fall_probability_threshold,
        "fall_confirmed": fall_confirmed,
    }
