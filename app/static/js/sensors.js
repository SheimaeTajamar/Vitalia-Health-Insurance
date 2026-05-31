const permissionBtn = document.getElementById("permissionBtn");
const toggleBtn = document.getElementById("toggleBtn");
const statusText = document.getElementById("statusText");
const envText = document.getElementById("envText");
const installBtn = document.getElementById("installBtn");

const accChart = document.getElementById("accChart");
const gyroChart = document.getElementById("gyroChart");
const actionIcon = document.getElementById("actionIcon");
const actionLabel = document.getElementById("actionLabel");
const inferenceDetails = document.getElementById("inferenceDetails");

const metricAccTotal = document.getElementById("metricAccTotal");
const metricGyroTotal = document.getElementById("metricGyroTotal");
const metricSamples = document.getElementById("metricSamples");
const metricDuration = document.getElementById("metricDuration");
const metricModelStatus = document.getElementById("metricModelStatus");

const SAMPLE_RATE_FALLBACK = 50;
const RENDER_MAX_HZ = 60;
const HISTORY_WINDOW_SEC = 30;
const SENSOR_BUFFER_WINDOW_MS = 30 * 1000;
const MIN_INFERENCE_DATA_MS = 2 * 1000;
const INFERENCE_INTERVAL_MS = 1000;
const ACTIVITY_SMOOTHING_SIZE = 5;

let readingActive = false;
let permissionGranted = false;
let deferredPrompt = null;
let lastRender = 0;
let noDataTimer = null;
let inferenceInFlight = false;
let backendModelReady = false;

const history = [];
const maxHistoryPoints = 8000;
const sensorBuffer = [];
const activityPredHistory = [];

const latest = {
  acc: { x: 0, y: 0, z: 0 },
  gyro: { x: 0, y: 0, z: 0 },
  accTotal: 0,
  gyroTotal: 0,
};

const host = window.location.hostname;
const isLocalHost = host === "localhost" || host === "127.0.0.1" || host === "::1";
const secureForSensors = window.isSecureContext || isLocalHost;
const hasMotionApi = typeof DeviceMotionEvent !== "undefined";

const STATE_VISUAL = {
  parado: { icon: "/static/icons/person-standing.svg", label: "Quieto", className: "state-idle" },
  sentado: { icon: "/static/icons/person-fill-down.svg", label: "Sentado", className: "state-sitting" },
  caminando: { icon: "/static/icons/person-walking.svg", label: "Caminando", className: "state-walking" },
  trotando: { icon: "/static/icons/lightning-fill.svg", label: "Corriendo", className: "state-running" },
  subiendo_escaleras: {
    icon: "/static/icons/person-fill-up.svg",
    label: "Subiendo escaleras",
    className: "state-stairs-up",
  },
  bajando_escaleras: {
    icon: "/static/icons/person-fill-down.svg",
    label: "Bajando escaleras",
    className: "state-stairs-down",
  },
  posible_caida_en_revision: {
    icon: "/static/icons/exclamation-triangle-fill.svg",
    label: "Posible caida en revision",
    className: "state-warning",
  },
  caida_detectada: {
    icon: "/static/icons/exclamation-triangle-fill.svg",
    label: "Caida detectada",
    className: "state-danger",
  },
  esperando_mas_datos: {
    icon: "/static/icons/person-standing.svg",
    label: "Esperando mas datos",
    className: "state-loading",
  },
  modelo_no_disponible: {
    icon: "/static/icons/exclamation-triangle-fill.svg",
    label: "Modelo no disponible",
    className: "state-error",
  },
};

const STATE_CLASSES = [
  "state-idle",
  "state-sitting",
  "state-walking",
  "state-running",
  "state-stairs-up",
  "state-stairs-down",
  "state-warning",
  "state-danger",
  "state-loading",
  "state-error",
];

function setStatus(text) {
  if (statusText) {
    statusText.textContent = text;
  }
}

function setModelStatus(text) {
  if (metricModelStatus) {
    metricModelStatus.textContent = text;
  }
}

function applyStateClass(stateKey) {
  const visual = STATE_VISUAL[stateKey] || STATE_VISUAL.esperando_mas_datos;
  document.body.classList.remove(...STATE_CLASSES);
  document.body.classList.add(visual.className);
}

function setAction(stateKey, customLabel = null) {
  const visual = STATE_VISUAL[stateKey] || STATE_VISUAL.esperando_mas_datos;
  actionIcon.src = visual.icon;
  actionLabel.textContent = customLabel || visual.label;
  applyStateClass(stateKey);
}

function setEnvironmentText() {
  if (!envText) {
    return;
  }
  if (!secureForSensors) {
    envText.textContent = "Entorno: usa HTTPS para habilitar sensores en movil.";
    return;
  }
  if (!hasMotionApi) {
    envText.textContent = "Entorno: este navegador no soporta DeviceMotionEvent.";
    return;
  }
  envText.textContent = "Entorno: navegador compatible para lectura de sensores.";
}

function updateMetrics() {
  if (metricAccTotal) {
    metricAccTotal.textContent = latest.accTotal.toFixed(2);
  }
  if (metricGyroTotal) {
    metricGyroTotal.textContent = latest.gyroTotal.toFixed(2);
  }
  if (metricSamples) {
    metricSamples.textContent = String(sensorBuffer.length);
  }
  if (metricDuration) {
    metricDuration.textContent = `${bufferDurationSeconds().toFixed(1)}s`;
  }
}

function bufferDurationSeconds() {
  if (sensorBuffer.length < 2) {
    return 0;
  }
  return Math.max(0, (sensorBuffer[sensorBuffer.length - 1].ts_ms - sensorBuffer[0].ts_ms) / 1000);
}

function pushHistory(tsMs) {
  history.push({
    tsMs,
    accVal: Number(latest.accTotal),
    gyroVal: Number(latest.gyroTotal),
  });

  const minTs = tsMs - HISTORY_WINDOW_SEC * 1000;
  while (history.length > 0 && history[0].tsMs < minTs) {
    history.shift();
  }
  if (history.length > maxHistoryPoints) {
    history.splice(0, history.length - maxHistoryPoints);
  }
}

function pushSensorSample(tsMs) {
  sensorBuffer.push({
    ts_ms: tsMs,
    acc_x: Number(latest.acc.x || 0),
    acc_y: Number(latest.acc.y || 0),
    acc_z: Number(latest.acc.z || 0),
    gyr_x: Number(latest.gyro.x || 0),
    gyr_y: Number(latest.gyro.y || 0),
    gyr_z: Number(latest.gyro.z || 0),
  });

  const minTs = tsMs - SENSOR_BUFFER_WINDOW_MS;
  while (sensorBuffer.length > 0 && sensorBuffer[0].ts_ms < minTs) {
    sensorBuffer.shift();
  }
}

function setupCanvas(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(320, Math.floor(canvas.clientWidth));
  const height = Math.max(180, Math.floor(canvas.clientHeight));
  canvas.width = Math.floor(width * ratio);
  canvas.height = Math.floor(height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { ctx, width, height };
}

function drawChart(canvas, key, color, label, yMin, yMax) {
  const { ctx, width, height } = setupCanvas(canvas);
  ctx.clearRect(0, 0, width, height);

  const plotLeft = 38;
  const plotRight = width - 8;
  const plotTop = 10;
  const plotBottom = height - 24;
  const plotWidth = plotRight - plotLeft;
  const plotHeight = plotBottom - plotTop;

  ctx.strokeStyle = "rgba(200, 220, 240, 0.25)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(plotLeft, plotTop);
  ctx.lineTo(plotLeft, plotBottom);
  ctx.lineTo(plotRight, plotBottom);
  ctx.stroke();

  const zeroY = plotBottom - ((0 - yMin) / (yMax - yMin)) * plotHeight;
  if (zeroY >= plotTop && zeroY <= plotBottom) {
    ctx.strokeStyle = "rgba(200, 220, 240, 0.22)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(plotLeft, zeroY);
    ctx.lineTo(plotRight, zeroY);
    ctx.stroke();
  }

  if (history.length < 2) {
    ctx.fillStyle = "rgba(234, 244, 255, 0.85)";
    ctx.font = "12px Segoe UI";
    ctx.fillText("Esperando datos...", 46, Math.floor(height / 2));
    return;
  }

  const t0 = history[0].tsMs;
  const t1 = history[history.length - 1].tsMs;
  const span = Math.max(1, t1 - t0);

  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  history.forEach((point, idx) => {
    const tx = (point.tsMs - t0) / span;
    const clamped = Math.max(yMin, Math.min(yMax, point[key]));
    const ty = (clamped - yMin) / (yMax - yMin);
    const x = plotLeft + tx * plotWidth;
    const y = plotBottom - ty * plotHeight;
    if (idx === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });
  ctx.stroke();

  ctx.fillStyle = "rgba(234, 244, 255, 0.95)";
  ctx.font = "12px Segoe UI";
  ctx.fillText(label, plotLeft, 22);
  ctx.fillText("30s", plotRight - 24, plotBottom + 14);
}

function renderCharts() {
  drawChart(accChart, "accVal", "#48d1cc", "Aceleracion", -50, 50);
  drawChart(gyroChart, "gyroVal", "#d4af37", "Giroscopio", -500, 500);
}

function renderData(timestampMs) {
  const minGapMs = 1000 / RENDER_MAX_HZ;
  if (timestampMs - lastRender < minGapMs) {
    return;
  }
  lastRender = timestampMs;
  pushHistory(timestampMs);
  pushSensorSample(timestampMs);
  updateMetrics();
  renderCharts();
}

function safeNumber(value) {
  return Number.isFinite(value) ? Number(value) : 0;
}

function onMotion(event) {
  const tsMs = Date.now();
  const acc = event.acceleration || event.accelerationIncludingGravity;
  const rot = event.rotationRate;

  if (!acc && !rot) {
    return;
  }

  if (acc) {
    latest.acc.x = safeNumber(acc.x);
    latest.acc.y = safeNumber(acc.y);
    latest.acc.z = safeNumber(acc.z);
  }
  if (rot) {
    latest.gyro.x = safeNumber(rot.alpha);
    latest.gyro.y = safeNumber(rot.beta);
    latest.gyro.z = safeNumber(rot.gamma);
  }

  latest.accTotal = Math.sqrt(latest.acc.x ** 2 + latest.acc.y ** 2 + latest.acc.z ** 2);
  latest.gyroTotal = Math.sqrt(latest.gyro.x ** 2 + latest.gyro.y ** 2 + latest.gyro.z ** 2);
  renderData(tsMs);
}

function estimateSampleRateHz(samples) {
  if (samples.length < 6) {
    return SAMPLE_RATE_FALLBACK;
  }

  const recent = samples.slice(-40);
  const deltas = [];
  for (let i = 1; i < recent.length; i += 1) {
    const delta = recent[i].ts_ms - recent[i - 1].ts_ms;
    if (delta > 0 && delta < 500) {
      deltas.push(delta);
    }
  }

  if (deltas.length < 4) {
    return SAMPLE_RATE_FALLBACK;
  }

  deltas.sort((a, b) => a - b);
  const median = deltas[Math.floor(deltas.length / 2)];
  if (!Number.isFinite(median) || median <= 0) {
    return SAMPLE_RATE_FALLBACK;
  }

  const hz = 1000 / median;
  return Math.min(120, Math.max(10, Number(hz.toFixed(2))));
}

function mode(items) {
  if (items.length === 0) {
    return null;
  }
  const count = new Map();
  let best = items[items.length - 1];
  let bestCount = 0;
  for (const item of items) {
    const next = (count.get(item) || 0) + 1;
    count.set(item, next);
    if (next >= bestCount) {
      bestCount = next;
      best = item;
    }
  }
  return best;
}

function smoothedActivity(rawActivity) {
  if (!rawActivity) {
    return "esperando_mas_datos";
  }
  activityPredHistory.push(rawActivity);
  while (activityPredHistory.length > ACTIVITY_SMOOTHING_SIZE) {
    activityPredHistory.shift();
  }
  return mode(activityPredHistory) || rawActivity;
}

function formatPercent(value) {
  if (!Number.isFinite(value)) {
    return "--%";
  }
  return `${(value * 100).toFixed(1)}%`;
}

function updateInferenceDetails(data) {
  if (!inferenceDetails) {
    return;
  }

  const confidence = formatPercent(Number(data.activity_confidence));
  const fallProbability = data.fall_probability == null ? "--%" : formatPercent(Number(data.fall_probability));
  const peak = Number.isFinite(Number(data.fall_peak_value)) ? Number(data.fall_peak_value).toFixed(2) : "--";
  const rnn = data.rnn_evaluated ? "Si" : "No";
  inferenceDetails.textContent = `Conf: ${confidence} | Caida: ${fallProbability} | Pico: ${peak} | RNN: ${rnn}`;
}

function resolveStateForDisplay(data) {
  const responseState = data.state || "esperando_mas_datos";
  if (responseState === "caida_detectada" || data.fall_confirmed) {
    return "caida_detectada";
  }
  if (responseState === "posible_caida_en_revision") {
    return "posible_caida_en_revision";
  }
  if (responseState === "modelo_no_disponible") {
    return "modelo_no_disponible";
  }
  if (responseState === "esperando_mas_datos") {
    return "esperando_mas_datos";
  }

  return smoothedActivity(data.activity_label || responseState);
}

async function fetchModelStatus() {
  try {
    const response = await fetch("/api/model-status");
    const data = await response.json();
    backendModelReady = Boolean(data.ready);
    if (backendModelReady) {
      setModelStatus("Listo");
      setStatus("Modelo IA listo.");
    } else {
      const errorText = data.error || "sin detalle";
      setModelStatus("No disponible");
      setStatus(`Modelo no disponible: ${errorText}`);
      setAction("modelo_no_disponible");
    }
  } catch (error) {
    backendModelReady = false;
    setModelStatus("Error de red");
    setStatus("Error de red consultando /api/model-status.");
    setAction("modelo_no_disponible");
  }
}

async function runInference() {
  if (!readingActive || !backendModelReady || inferenceInFlight) {
    return;
  }
  if (sensorBuffer.length < 2 || bufferDurationSeconds() < 2) {
    setAction("esperando_mas_datos");
    return;
  }

  inferenceInFlight = true;
  try {
    const sampleRateHz = estimateSampleRateHz(sensorBuffer);
    const payload = {
      sample_rate_hz: sampleRateHz,
      samples: sensorBuffer.slice(),
    };

    const response = await fetch("/api/infer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    if (!data.ready) {
      setModelStatus("No disponible");
      setStatus(`Modelo no disponible: ${data.error || "sin detalle"}`);
      setAction("modelo_no_disponible");
      return;
    }

    const uiState = resolveStateForDisplay(data);
    setAction(uiState);
    setStatus(`Estado actual: ${STATE_VISUAL[uiState]?.label || uiState}`);
    updateInferenceDetails(data);
  } catch (error) {
    setStatus("Error de inferencia");
    setAction("modelo_no_disponible", "Error de inferencia");
  } finally {
    inferenceInFlight = false;
  }
}

function startReading() {
  history.length = 0;
  sensorBuffer.length = 0;
  activityPredHistory.length = 0;
  window.clearTimeout(noDataTimer);
  window.addEventListener("devicemotion", onMotion);

  readingActive = true;
  toggleBtn.textContent = "Detener lectura";
  setStatus("Lectura activa.");
  setAction("esperando_mas_datos");
  updateMetrics();
  renderCharts();

  noDataTimer = window.setTimeout(() => {
    if (readingActive && sensorBuffer.length === 0) {
      setStatus("Sensores no soportados o sin datos. Revisa permisos del navegador.");
      setAction("modelo_no_disponible", "Sin datos de sensores");
    }
  }, 3000);
}

function stopReading() {
  window.removeEventListener("devicemotion", onMotion);
  window.clearTimeout(noDataTimer);
  readingActive = false;
  toggleBtn.textContent = "Iniciar lectura";
  setStatus("Lectura detenida.");
}

async function requestSensorPermissions() {
  if (!secureForSensors) {
    setStatus("Error de permisos: contexto inseguro. Usa HTTPS.");
    return;
  }
  if (!hasMotionApi) {
    setStatus("Sensores no soportados en este navegador.");
    return;
  }

  try {
    if (typeof DeviceMotionEvent.requestPermission === "function") {
      const motion = await DeviceMotionEvent.requestPermission();
      if (motion !== "granted") {
        throw new Error("permiso de movimiento denegado");
      }
    }

    permissionGranted = true;
    toggleBtn.disabled = false;
    setStatus("Permisos concedidos.");
  } catch (error) {
    const message = error instanceof Error ? error.message : "error desconocido";
    setStatus(`Error de permisos: ${message}`);
  }
}

permissionBtn.addEventListener("click", requestSensorPermissions);

toggleBtn.addEventListener("click", () => {
  if (!permissionGranted) {
    setStatus("Concede permisos antes de iniciar.");
    return;
  }
  if (readingActive) {
    stopReading();
  } else {
    startReading();
  }
});

window.addEventListener("beforeinstallprompt", (event) => {
  event.preventDefault();
  deferredPrompt = event;
  installBtn.hidden = false;
});

installBtn.addEventListener("click", async () => {
  if (!deferredPrompt) {
    return;
  }
  deferredPrompt.prompt();
  await deferredPrompt.userChoice;
  deferredPrompt = null;
  installBtn.hidden = true;
});

if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("/static/sw.js").catch(() => {
      setStatus("Servicio offline no disponible.");
    });
  });
}

window.addEventListener("resize", () => renderCharts());
setInterval(() => {
  void runInference();
}, INFERENCE_INTERVAL_MS);

setEnvironmentText();
setAction("esperando_mas_datos");
setModelStatus("Cargando...");
updateMetrics();
renderCharts();
void fetchModelStatus();
