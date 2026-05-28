const permissionBtn = document.getElementById("permissionBtn");
const toggleBtn = document.getElementById("toggleBtn");
const statusText = document.getElementById("statusText");
const envText = document.getElementById("envText");
const installBtn = document.getElementById("installBtn");

const accChart = document.getElementById("accChart");
const gyroChart = document.getElementById("gyroChart");
const actionIcon = document.getElementById("actionIcon");
const actionLabel = document.getElementById("actionLabel");

const SAMPLE_HZ = 60;
const HISTORY_WINDOW_SEC = 30;

let readingActive = false;
let permissionGranted = false;
let deferredPrompt = null;
let lastRender = 0;
let eventCount = 0;
let noDataTimer = null;
let hasRotationRateData = false;

let backendModelReady = false;
let inferenceInFlight = false;

const history = [];
const maxHistoryPoints = 8000;
const modelBuffer = [];
const modelBufferMaxMs = 40 * 1000;

const latest = {
  acc: { x: 0, y: 0, z: 0 },
  gyro: { alpha: 0, beta: 0, gamma: 0 },
};

const host = window.location.hostname;
const isLocalHost = host === "localhost" || host === "127.0.0.1" || host === "::1";
const secureForSensors = window.isSecureContext || isLocalHost;
const hasMotionApi = typeof DeviceMotionEvent !== "undefined";
const hasOrientationApi = typeof DeviceOrientationEvent !== "undefined";

const STATE_VISUAL = {
  caminando: { icon: "/static/icons/person-walking.svg", label: "Caminando" },
  trotando: { icon: "/static/icons/lightning-fill.svg", label: "Trotando" },
  parado: { icon: "/static/icons/person-standing.svg", label: "Parado" },
  sentado: { icon: "/static/icons/person-fill-down.svg", label: "Sentado" },
  subiendo_escaleras: { icon: "/static/icons/person-fill-up.svg", label: "Subiendo escaleras" },
  bajando_escaleras: { icon: "/static/icons/person-fill-down.svg", label: "Bajando escaleras" },
  posible_caida_en_revision: { icon: "/static/icons/exclamation-triangle-fill.svg", label: "Posible caida" },
  caida_detectada: { icon: "/static/icons/exclamation-triangle-fill.svg", label: "Caida detectada" },
  esperando_mas_datos: { icon: "/static/icons/person-standing.svg", label: "Esperando datos" },
  modelo_no_disponible: { icon: "/static/icons/exclamation-triangle-fill.svg", label: "Modelo no disponible" },
};

function setStatus(text) {
  if (statusText) {
    statusText.textContent = `Estado: ${text}`;
  }
}

function setAction(stateKey) {
  const visual = STATE_VISUAL[stateKey] || STATE_VISUAL.esperando_mas_datos;
  actionIcon.src = visual.icon;
  actionLabel.textContent = visual.label;
}

function setEnvironmentText() {
  if (!envText) {
    return;
  }
  if (!secureForSensors) {
    envText.textContent =
      "Entorno: contexto no seguro. Abre la app con HTTPS para habilitar sensores en la mayoria de moviles.";
    return;
  }
  if (!hasMotionApi && !hasOrientationApi) {
    envText.textContent = "Entorno: este navegador no expone APIs de sensores.";
    return;
  }
  envText.textContent = "Entorno: navegador compatible. Puedes solicitar permisos.";
}

function pushHistory(tsMs) {
  const accRms = Math.sqrt(
    ((latest.acc.x || 0) ** 2 + (latest.acc.y || 0) ** 2 + (latest.acc.z || 0) ** 2) / 3,
  );
  const gyroRms = Math.sqrt(
    ((latest.gyro.alpha || 0) ** 2 + (latest.gyro.beta || 0) ** 2 + (latest.gyro.gamma || 0) ** 2) / 3,
  );

  history.push({
    tsMs,
    accVal: Number(accRms),
    gyroVal: Number(gyroRms),
  });

  const minTs = tsMs - HISTORY_WINDOW_SEC * 1000;
  while (history.length > 0 && history[0].tsMs < minTs) {
    history.shift();
  }
  if (history.length > maxHistoryPoints) {
    history.splice(0, history.length - maxHistoryPoints);
  }
}

function pushModelSample(tsMs) {
  modelBuffer.push({
    ts_ms: tsMs,
    acc_x: Number(latest.acc.x || 0),
    acc_y: Number(latest.acc.y || 0),
    acc_z: Number(latest.acc.z || 0),
    gyr_x: Number(latest.gyro.alpha || 0),
    gyr_y: Number(latest.gyro.beta || 0),
    gyr_z: Number(latest.gyro.gamma || 0),
  });

  const minTs = tsMs - modelBufferMaxMs;
  while (modelBuffer.length > 0 && modelBuffer[0].ts_ms < minTs) {
    modelBuffer.shift();
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
  ctx.fillText(`${label}: ${history[history.length - 1][key].toFixed(2)}`, plotLeft, 22);
  ctx.fillText(`rango ${yMin}..${yMax}`, plotLeft, 36);
  ctx.fillText("30s", plotRight - 24, plotBottom + 14);
}

function renderCharts() {
  drawChart(accChart, "accVal", "#48d1cc", "Acc RMS", -50, 50);
  drawChart(gyroChart, "gyroVal", "#d4af37", "Gyro RMS", -500, 500);
}

function renderData(timestampMs) {
  const minGapMs = 1000 / SAMPLE_HZ;
  if (timestampMs - lastRender < minGapMs) {
    return;
  }
  lastRender = timestampMs;
  pushHistory(timestampMs);
  pushModelSample(timestampMs);
  renderCharts();
}

function onMotion(event) {
  eventCount += 1;
  const acc = event.acceleration || event.accelerationIncludingGravity;
  if (acc) {
    latest.acc.x = acc.x;
    latest.acc.y = acc.y;
    latest.acc.z = acc.z;
  }
  if (event.rotationRate) {
    hasRotationRateData = true;
    latest.gyro.alpha = event.rotationRate.alpha;
    latest.gyro.beta = event.rotationRate.beta;
    latest.gyro.gamma = event.rotationRate.gamma;
  }
  renderData(Date.now());
}

function onOrientation(event) {
  eventCount += 1;
  if (!hasRotationRateData) {
    latest.gyro.alpha = event.alpha;
    latest.gyro.beta = event.beta;
    latest.gyro.gamma = event.gamma;
  }
  renderData(Date.now());
}

async function fetchModelStatus() {
  try {
    const response = await fetch("/api/model-status");
    const data = await response.json();
    backendModelReady = Boolean(data.ready);
    if (backendModelReady) {
      setStatus("modelo IA cargado.");
    } else {
      setStatus(`modelo no disponible: ${data.error || "sin detalle"}`);
      setAction("modelo_no_disponible");
    }
  } catch (error) {
    backendModelReady = false;
    setStatus("sin conexion al backend de inferencia.");
    setAction("modelo_no_disponible");
  }
}

function latestSamplesForInference(secondsBack = 25) {
  if (modelBuffer.length === 0) {
    return [];
  }
  const maxTs = modelBuffer[modelBuffer.length - 1].ts_ms;
  const minTs = maxTs - secondsBack * 1000;
  return modelBuffer.filter((s) => s.ts_ms >= minTs);
}

async function runInference() {
  if (!readingActive || !backendModelReady || inferenceInFlight) {
    return;
  }

  const samples = latestSamplesForInference(25);
  if (samples.length < 20) {
    return;
  }

  inferenceInFlight = true;
  try {
    const response = await fetch("/api/infer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ samples, sample_rate_hz: SAMPLE_HZ }),
    });
    const data = await response.json();

    if (!data.ready) {
      setAction("modelo_no_disponible");
      setStatus(`modelo no disponible: ${data.error || "sin detalle"}`);
      return;
    }

    setAction(data.state || data.activity_label || "esperando_mas_datos");
    setStatus(`estado IA: ${data.state || data.activity_label || "sin estado"}`);
  } catch (error) {
    setStatus("error de inferencia.");
    setAction("modelo_no_disponible");
  } finally {
    inferenceInFlight = false;
  }
}

function startReading() {
  eventCount = 0;
  hasRotationRateData = false;
  history.length = 0;
  modelBuffer.length = 0;
  window.addEventListener("devicemotion", onMotion);
  window.addEventListener("deviceorientation", onOrientation);
  readingActive = true;
  toggleBtn.textContent = "Detener lectura";
  setStatus(`capturando sensores a ${SAMPLE_HZ} Hz.`);

  noDataTimer = window.setTimeout(() => {
    if (readingActive && eventCount === 0) {
      setStatus("sin eventos de sensores. Revisa HTTPS, permisos y ajustes del navegador.");
    }
  }, 3000);
}

function stopReading() {
  window.removeEventListener("devicemotion", onMotion);
  window.removeEventListener("deviceorientation", onOrientation);
  window.clearTimeout(noDataTimer);
  readingActive = false;
  toggleBtn.textContent = "Iniciar lectura";
  setStatus("lectura detenida.");
}

async function requestSensorPermissions() {
  if (!secureForSensors) {
    setStatus("sensores bloqueados por contexto inseguro. Usa HTTPS.");
    return;
  }
  if (!hasMotionApi && !hasOrientationApi) {
    setStatus("este navegador no soporta sensores.");
    return;
  }

  try {
    if (hasMotionApi && typeof DeviceMotionEvent.requestPermission === "function") {
      const motion = await DeviceMotionEvent.requestPermission();
      if (motion !== "granted") {
        throw new Error("permiso de movimiento denegado");
      }
    }
    if (hasOrientationApi && typeof DeviceOrientationEvent.requestPermission === "function") {
      const orientation = await DeviceOrientationEvent.requestPermission();
      if (orientation !== "granted") {
        throw new Error("permiso de orientacion denegado");
      }
    }

    permissionGranted = true;
    toggleBtn.disabled = false;
    setStatus("permisos listos.");
  } catch (error) {
    const message = error instanceof Error ? error.message : "error desconocido";
    setStatus(`error de permisos: ${message}`);
  }
}

permissionBtn.addEventListener("click", requestSensorPermissions);

toggleBtn.addEventListener("click", () => {
  if (!permissionGranted) {
    setStatus("concede permisos antes de iniciar.");
    return;
  }
  if (readingActive) {
    stopReading();
    return;
  }
  startReading();
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
      setStatus("servicio offline no disponible.");
    });
  });
}

window.addEventListener("resize", () => renderCharts());
setInterval(() => {
  void runInference();
}, 1000);

setEnvironmentText();
setAction("esperando_mas_datos");
renderCharts();
void fetchModelStatus();
