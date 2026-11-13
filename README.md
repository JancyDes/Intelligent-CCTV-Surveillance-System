# Sentinel AI CCTV dashboard

## Run locally

In PowerShell, from this folder:

```powershell
npx serve .
```

Then open the localhost address shown in the terminal. Camera access requires `localhost` (or HTTPS) and permission in the browser. On a phone, use HTTPS or host the project on the same network with a trusted certificate.

## Connect your Colab model

This interface currently demonstrates the detection flow in the browser. To use the model trained in Colab, deploy it behind an HTTP endpoint (for example FastAPI, Flask, or a hosted inference service). The endpoint should accept a video frame or video file and return activity detections such as:

```json
{"detections":[{"label":"fight","confidence":0.94,"timestamp":12.8}]}
```

Replace the simulated result creation in `addAlert()` / `analyzeUpload()` in `app.js` with a `fetch()` call to that endpoint, then convert returned labels to the `activities` objects at the top of the file.
