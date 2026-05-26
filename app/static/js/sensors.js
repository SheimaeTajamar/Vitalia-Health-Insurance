const permissionBtn = document.getElementById("permissionBtn");
const toggleBtn = document.getElementById("toggleBtn");
const sampleRate = document.getElementById("sampleRate");
const rateLabel = document.getElementById("rateLabel");
const statusText = document.getElementById("statusText");
const envText = document.getElementById("envText");
const installBtn = document.getElementById("installBtn");

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
  // Fallback when rotationRate is unavailable: orientation angles.
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
  window.addEventListener("devicemotion", onMotion);
  window.addEventListener("deviceorientation", onOrientation);
  readingActive = true;
  toggleBtn.textContent = "Detener lectura";
  setStatus(`capturando sensores a ${sampleHz} Hz.`);

  noDataTimer = window.setTimeout(() => {
    if (readingActive && eventCount === 0) {
      setStatus(
        "sin eventos de sensores. Revisa HTTPS, permisos del sitio y que el navegador permita sensores de movimiento.",
      );
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
    // iOS Safari requires explicit permission inside a user gesture.
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
    setStatus("permisos listos. Si no aparece popup, es normal en Android/Chrome.");
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

setEnvironmentText();
