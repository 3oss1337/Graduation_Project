package com.furnishar.mobile

import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel

class MainActivity : FlutterActivity() {
    private val channelName = "furnishar/onnx"

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, channelName)
            .setMethodCallHandler { call, result ->
                when (call.method) {
                    "hasModels" -> result.success(hasBundledModels())
                    "runPipeline" -> {
                        result.error(
                            "onnx_not_configured",
                            "Android ONNX Runtime sessions are scaffolded but not wired. " +
                                "Add onnxruntime-android and implement MobileSAM, rembg, and CLIP image/text inference here.",
                            null
                        )
                    }
                    else -> result.notImplemented()
                }
            }
    }

    private fun hasBundledModels(): Boolean {
        val required = listOf(
            "flutter_assets/assets/models/mobile_sam_encoder.onnx",
            "flutter_assets/assets/models/mobile_sam_decoder.onnx",
            "flutter_assets/assets/models/rembg.onnx",
            "flutter_assets/assets/models/clip_image_encoder.onnx",
            "flutter_assets/assets/models/clip_text_embeddings.json"
        )
        return required.all { asset ->
            try {
                assets.open(asset).close()
                true
            } catch (_: Exception) {
                false
            }
        }
    }
}
