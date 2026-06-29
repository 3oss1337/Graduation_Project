import 'package:flutter/services.dart';

import '../models/vision_result.dart';

class OnDeviceVisionService {
  OnDeviceVisionService({MethodChannel? channel})
      : _channel = channel ?? const MethodChannel('furnishar/onnx');

  final MethodChannel _channel;

  Future<bool> hasModels() async {
    final result = await _channel.invokeMethod<bool>('hasModels');
    return result ?? false;
  }

  Future<VisionResult> runPipeline({
    required String imagePath,
    required bool segmentationEnabled,
  }) async {
    final result = await _channel.invokeMapMethod<String, dynamic>(
      'runPipeline',
      {
        'imagePath': imagePath,
        'segmentationEnabled': segmentationEnabled,
      },
    );

    if (result == null) {
      throw PlatformException(
        code: 'empty_result',
        message: 'Native ONNX pipeline returned no result.',
      );
    }

    return VisionResult.fromMap(result);
  }
}
