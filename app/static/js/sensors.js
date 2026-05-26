const permissionBtn = document.getElementById("permissionBtn");
const toggleBtn = document.getElementById("toggleBtn");
const sampleRate = document.getElementById("sampleRate");
const rateLabel = document.getElementById("rateLabel");
const statusText = document.getElementById("statusText");
const envText = document.getElementById("envText");
const installBtn = document.getElementById("installBtn");

const historyWindow = document.getElementById("historyWindow");
const historyLabel = document.getElementById("historyLabel");
const accChart = document.getElementById("accChart");
const gyroChart = document.getElementById("gyroChart");

const accX = document.getElementById("accX");
const accY = document.getElementById("accY");
const accZ = document.getElementById("accZ");
const gyroA = document.getElementById("gyroA");
const gyroB = document.getElementById("gyroB");
const gyroG = document.getElementById("gyroG");

let readingActive = false;
let permissionGranted = false;
let deferredPrompt = null;
let sampleHz = Number(sampleRate.value);
let lastRender = 0;
let eventCount = 0;
let noDataTimer = null;
let hasRotationRateData = false;
let historyWindowSec = Number(historyWindow.value);

const history = [];
const maxHistoryPoints = 6000;

const latest = {
  acc: { x: 0, y: 0, z: 0 },
  gyro: { alpha: 0, beta: 0, gamma: 0 },
};

const host = window.location.hostname;
const isLocalHost = host === "localhost" || host === "127.0.0.1" || host === "::1";
const secureForSensors = window.isSecureContext || isLocalHost;
const hasMotionApi = typeof DeviceMotionEvent !== "undefined";
const hasOrientationApi = typeof DeviceOrientationEvent !== "undefined";

const toFixed = (n) => Number(n || 0).toFixed(2);

function setStatus(text) {
  statusText.textContent = `Estado: ${text}`;
}

function setEnvironmentText() {
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

function accMagnitude() {
  return Math.sqrt((latest.acc.x || 0) ** 2 + (latest.acc.y || 0) ** 2 + (latest.acc.z || 0) ** 2);
}

function gyroMagnitude() {
  return Math.sqrt(
    (latest.gyro.alpha || 0) ** 2 + (latest.gyro.beta || 0) ** 2 + (latest.gyro.gamma || 0) ** 2,
  );
}

function pushHistory(tsMs) {
  history.push({
    tsMs,
    accMag: accMagnitude(),
    gyroMag: gyroMagnitude(),
  });

  const minTs = tsMs - historyWindowSec * 1000;
  while (history.length > 0 && history[0].tsMs < minTs) {
    history.shift();
  }
  if (history.length > maxHistoryPoints) {
    history.splice(0, history.length - maxHistoryPoints);
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

function drawChart(canvas, key, color, label) {
  const { ctx, width, height } = setupCanvas(canvas);

  ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = "rgba(200, 220, 240, 0.25)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(38, 10);
  ctx.lineTo(38, height - 24);
  ctx.lineTo(width - 8, height - 24);
  ctx.stroke();

  if (history.length < 2) {
    ctx.fillStyle = "rgba(234, 244, 255, 0.85)";
    ctx.font = "12px Segoe UI";
    ctx.fillText("Esperando datos...", 46, Math.floor(height / 2));
    return;
  }

  let maxVal = 0;
  for (const point of history) {
    if (point[key] > maxVal) {
      maxVal = point[key];
    }
  }
  const minVal = 0;
  const yMax = maxVal > 0.01 ? maxVal * 1.1 : 1;

  const t0 = history[0].tsMs;
  const t1 = history[history.length - 1].tsMs;
  const span = Math.max(1, t1 - t0);

  const plotLeft = 38;
  const plotRight = width - 8;
  const plotTop = 10;
  const plotBottom = height - 24;
  const plotWidth = plotRight - plotLeft;
  const plotHeight = plotBottom - plotTop;

  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  history.forEach((point, idx) => {
    const tx = (point.tsMs - t0) / span;
    const ty = (point[key] - minVal) / (yMax - minVal);
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
  ctx.fillText(`max ${yMax.toFixed(2)}`, plotLeft, 36);
  ctx.fillText(`${historyWindowSec}s`, plotRight - 34, plotBottom + 14);
}

function renderCharts() {
  drawChart(accChart, "accMag", "#48d1cc", "Acc");
  drawChart(gyroChart, "gyroMag", "#d4af37", "Gyro");
}

function renderData(timestampMs) {
  const minGapMs = 1000 / sampleHz;
  if (timestampMs - lastRender < minGapMs) {
    return;
  }
  lastRender = timestampMs;
  accX.textContent = toFixed(latest.acc.x);
  accY.textContent = toFixed(latest.acc.y);
  accZ.textContent = toFixed(latest.acc.z);
  gyroA.textContent = toFixed(latest.gyro.alpha);
  gyroB.textContent = toFixed(latest.gyro.beta);
  gyroG.textContent = toFixed(latest.gyro.gamma);
  pushHistory(timestampMs);
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

function startReading() {
  eventCount = 0;
  hasRotationRateData = false;
  history.length = 0;
  window.addEventListener("devicemotion", onMotion);
  window.addEventListener("deviceorientation", onOrientation);
  readingActive = true;
  toggleBtn.textContent = "Detener lectura";
  setStatus(`capturando sensores a ${sampleHz} Hz.`);

  noDataTimer = window.setTimeout(() => {
    if (readingActive && eventCount === 0) {
      setStatus("sin eventos de sensores. Revisa HTTPS, permisos del sitio y ajustes del navegador.");
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
    setStatus("permisos listos. Si no aparece popup, puede ser normal en Android/Chrome.");
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

sampleRate.addEventListener("input", (event) => {
  sampleHz = Number(event.target.value);
  rateLabel.textContent = String(sampleHz);
  if (readingActive) {
    setStatus(`capturando sensores a ${sampleHz} Hz.`);
  }
});

historyWindow.addEventListener("input", (event) => {
  historyWindowSec = Number(event.target.value);
  historyLabel.textContent = String(historyWindowSec);
  if (history.length > 0) {
    const now = history[history.length - 1].tsMs;
    const minTs = now - historyWindowSec * 1000;
    while (history.length > 0 && history[0].tsMs < minTs) {
      history.shift();
    }
  }
  renderCharts();
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

setEnvironmentText();
renderCharts();
