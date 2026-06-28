# FurnishAR Mobile

Flutter client for the FurnishAR mobile architecture.

The app captures an image, optionally runs MobileSAM, always runs rembg, classifies with MobileNetV4-small, uploads the processed object image and classification to `backend.py`, then displays the returned GLB in a viewer with AR handoff.

## Project Layout

- `lib/main.dart`: app entrypoint and main workflow UI.
- `lib/services/on_device_vision_service.dart`: method-channel bridge to Android/iOS ONNX Runtime.
- `lib/services/furnishar_api.dart`: multipart client for `POST /mobile/reconstruct`.
- `lib/models/`: typed pipeline and reconstruction models.
- `android/` and `ios/`: native method-channel stubs where ONNX Runtime Mobile integration belongs.
- `assets/models/`: place exported ONNX models here.

## Required Model Assets

Copy these files into `assets/models/`:

- `mobile_sam_encoder.onnx`
- `mobile_sam_decoder.onnx`
- `rembg.onnx`
- `mobilenetv4_small.onnx`

`assets/labels.json` contains the category names expected by `backend.py`.

## Run

Install Flutter, then from this folder:

```bash
flutter pub get
flutter run
```

This workspace did not have Flutter installed when the scaffold was created. If your local Flutter SDK reports missing generated Android/iOS project metadata, run:

```bash
flutter create --platforms=android,ios .
```

Then keep the existing `lib/`, `assets/`, and native method-channel code, or re-apply the method-channel snippets from `android/app/src/main/kotlin/com/furnishar/mobile/MainActivity.kt` and `ios/Runner/AppDelegate.swift`.

Set the backend URL in the app settings. For local Wi-Fi testing, use your computer's LAN address, for example:

```text
http://192.168.1.20:8000
```

Start the backend with standard `256^3` marching cubes:

```bash
python backend.py --port 8000 --mc-resolution 256
```

## Native ONNX Integration

The Dart side calls `MethodChannel('furnishar/onnx')` with:

- `runPipeline`
- `hasModels`

The native Android/iOS files are scaffolded and return a clear placeholder error until ONNX Runtime Mobile sessions are wired. Add ONNX Runtime dependencies and implement:

1. Decode the captured image.
2. If `segmentationEnabled`, run MobileSAM encoder/decoder and apply the mask.
3. Run rembg regardless.
4. Run MobileNetV4-small.
5. Return a processed PNG path plus category/confidence.

The Flutter app already consumes that result and sends it to the backend.
