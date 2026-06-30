# FurnishAR

FurnishAR is an AI furniture reconstruction and AR preview app. It turns a single furniture image into a textured 3D model that can be inspected, exported, and opened in AR.

## Features

- Single-image furniture reconstruction with TripoSR.
- Automatic background removal.
- CLIP-based furniture classification and LoRA adapter routing.
- Crowded-scene object selection with SAM segmentation.
- Textured GLB baking for AR and export.
- Main viewer modes: Textured, Solid, Wireframe, and Both.
- AR preview with `model-viewer`, Scene Viewer, Quick Look, and WebXR support.
- QR code mobile handoff for generated models.
- OBJ and GLB export.
- Optional adaptive marching cubes for faster mesh extraction.
- Backend warmup and CUDA mixed-precision/compile optimizations when available.
- Flutter mobile scaffold with camera/gallery capture, backend upload, model preview, and AR handoff.

## Project Layout

- `backend.py`: FastAPI backend for classification, segmentation, reconstruction, GLB baking, QR, AR viewer, and export.
- `FurnishAR.html` / `FurnishAR (1).html`: web frontend.
- `mobile/furnishar_mobile`: Flutter mobile app.
- `adapters/`: LoRA adapter files.
- `stabilityai/`: local TripoSR model config/weights.
- `static/models/`: generated GLB files served by the backend.

## Run The Backend

Install Python dependencies first:

```bash
pip install -r requirements.txt
```

Start the backend:

```bash
python backend.py --port 8000 --mc-resolution 256
```

Optional adaptive marching cubes:

```bash
python backend.py --port 8000 --adaptive --mc-resolution 256 --mc-coarse-resolution 128
```

Open the web app:

```text
http://localhost:8000/
```

Useful endpoints:

- `POST /classify`
- `POST /segment`
- `POST /reconstruct`
- `POST /mobile/reconstruct`
- `GET /viewer/{session_id}`
- `GET /qr/{session_id}`
- `GET /export?session_id=...&format=glb`

## Use The Web App

1. Start `backend.py`.
2. Open `http://localhost:8000/`.
3. Upload a furniture image.
4. Choose Single Object or Crowded Scene.
5. Keep Bake Texture enabled for the best GLB/AR result.
6. Generate the model.
7. Preview it, export OBJ/GLB, or scan the QR code for AR.

## Compile The Flutter App

Install Flutter and Android Studio first, then run:

```bash
cd mobile/furnishar_mobile
flutter pub get
flutter build apk --debug
```

For a release APK:

```bash
flutter build apk --release
```

To run on an emulator or connected device:

```bash
flutter run
```

If Flutter reports missing platform files:

```bash
flutter create --platforms=android,ios .
flutter pub get
```

Then keep the existing `lib/`, `assets/`, and native method-channel files.

## Mobile Backend URL

In the Flutter app settings, set the backend URL:

- Android emulator: `http://10.0.2.2:8000`
- Physical phone on the same Wi-Fi: `http://YOUR_PC_LAN_IP:8000`

Start the backend with:

```bash
python backend.py --port 8000 --mc-resolution 256
```

## Notes

- Existing generated GLB files do not update retroactively after backend changes; regenerate models to see new export behavior.

