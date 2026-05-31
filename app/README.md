# Vitalia Sensor App (FastAPI + Jinja2)

Aplicacion web para leer sensores de **aceleracion** y **giroscopio** desde movil, con:
- permisos de sensores desde UI
- tasa de muestreo configurable (Hz)
- instalacion como acceso directo (PWA)
- inferencia en backend con modelos `.keras` (CNN actividad + RNN caidas)

## 1. Requisitos

- Windows (o Linux/macOS)
- Python 3.10+ instalado
- `pip`

## 2. Instalacion

Desde la carpeta `app`:

```powershell
cd "C:\Users\Santy\Downloads\Tajamar Fight\app"
pip install -r requirements.txt
```

## 3. Ejecutar en local

```powershell
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Abrir en navegador:
- http://127.0.0.1:8000

## 4. Probar desde movil (misma WiFi)

Ejecutar servidor:

```powershell
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Ver IP local del PC:

```powershell
ipconfig
```

Abrir en movil:
- `http://TU_IP:8000` (ejemplo: `http://192.168.1.34:8000`)

## 5. Permitir puerto 8000 en firewall (Windows)

PowerShell como administrador:

```powershell
New-NetFirewallRule -DisplayName "FastAPI 8000 Inbound" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow -Profile Private
```

Verificar regla:

```powershell
Get-NetFirewallRule -DisplayName "FastAPI 8000 Inbound" | Get-NetFirewallPortFilter
```

## 6. Sensores no muestran datos: causa comun

Muchos navegadores moviles bloquean sensores en `http://IP_LOCAL`.

Opciones:
- usar Android por USB con `adb reverse` y abrir `http://localhost:8000` en el movil
- usar HTTPS temporal con Cloudflare Tunnel (sin desplegar codigo)

## 7. HTTPS temporal con Cloudflare Tunnel

1) Levantar FastAPI:

```powershell
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

2) Abrir tunel:

```powershell
cloudflared tunnel --url http://127.0.0.1:8000
```

3) Abrir en movil la URL `https://...trycloudflare.com` que imprime `cloudflared`.

## 8. Flujo de uso en la app

1. Pulsar `Conceder permisos`.
2. Pulsar `Iniciar lectura`.
3. Ajustar slider de Hz.
4. (Opcional) `Instalar acceso directo`.

## 9. Estructura

- `main.py`: servidor FastAPI + rutas
- `templates/index.html`: UI principal
- `static/js/sensors.js`: lectura de sensores y permisos
- `static/css/styles.css`: tema visual
- `static/manifest.webmanifest`: PWA manifest
- `static/sw.js`: service worker/cache

## 10. Solucion de problemas rapida

- Si no aparece popup de permisos en Android/Chrome, puede ser normal.
- Si estado indica contexto inseguro, abrir por HTTPS o localhost del propio movil.
- Si ves version vieja de la UI, borra datos del sitio o reinstala el acceso directo.

## 11. Integracion de modelos `.keras`

La app busca estos modelos automaticamente:

- `../model/modelo_cnn_actividad.keras`
- `../model/modelo_rnn_caidas.keras`

Endpoints nuevos:

- `GET /api/model-status` -> estado de carga de modelos
- `POST /api/infer` -> inferencia de actividad + logica de posible caida

Variables opcionales de ajuste:

- `CNN_WINDOW_SECONDS` (default `2.0`)
- `RNN_WINDOW_SECONDS` (default `30.0`)
- `FALL_PEAK_THRESHOLD` (default `3.0`)
- `FALL_PROBABILITY_THRESHOLD` (default `0.5`)
