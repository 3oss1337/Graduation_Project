import 'dart:convert';
import 'dart:io';

import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';

import '../models/reconstruction_result.dart';
import '../models/vision_result.dart';

class FurnishArApi {
  FurnishArApi({required this.baseUrl});

  final String baseUrl;

  Uri _uri(String path) {
    final normalized = baseUrl.endsWith('/')
        ? baseUrl.substring(0, baseUrl.length - 1)
        : baseUrl;
    return Uri.parse('$normalized$path');
  }

  String absoluteUrl(String pathOrUrl) {
    if (pathOrUrl.startsWith('http://') || pathOrUrl.startsWith('https://')) {
      return pathOrUrl;
    }
    final normalized = baseUrl.endsWith('/')
        ? baseUrl.substring(0, baseUrl.length - 1)
        : baseUrl;
    return '$normalized$pathOrUrl';
  }

  MediaType _imageMediaType(String path) {
    final extension = path.split('.').last.toLowerCase();
    return switch (extension) {
      'png' => MediaType('image', 'png'),
      'webp' => MediaType('image', 'webp'),
      'gif' => MediaType('image', 'gif'),
      'heic' => MediaType('image', 'heic'),
      'heif' => MediaType('image', 'heif'),
      _ => MediaType('image', 'jpeg'),
    };
  }

  Future<ReconstructionResult> reconstruct({
    required VisionResult vision,
    int resolution = 256,
    int textureResolution = 2048,
  }) async {
    final request = http.MultipartRequest(
      'POST',
      _uri('/mobile/reconstruct'),
    );

    request.files.add(
      await http.MultipartFile.fromPath(
        'file',
        vision.processedImagePath,
        filename: 'furnishar_object.png',
        contentType: _imageMediaType(vision.processedImagePath),
      ),
    );
    request.fields.addAll({
      'category': vision.category,
      'confidence': vision.confidence.toStringAsFixed(5),
      'segmentation': vision.segmentationUsed.toString(),
      'resolution': resolution.toString(),
      'texture_res': textureResolution.toString(),
      'include_geometry': 'false',
    });

    final streamed = await request.send().timeout(const Duration(minutes: 10));
    final response = await http.Response.fromStream(streamed);

    if (response.statusCode < 200 || response.statusCode >= 300) {
      throw HttpException(
        'Backend reconstruction failed (${response.statusCode}): ${response.body}',
        uri: _uri('/mobile/reconstruct'),
      );
    }

    return ReconstructionResult.fromJson(
      jsonDecode(response.body) as Map<String, dynamic>,
    );
  }

  Future<ReconstructionResult> reconstructWithBackendVision({
    required String imagePath,
    int resolution = 256,
    int textureResolution = 2048,
    bool removeBackground = true,
    bool segmentationEnabled = false,
  }) async {
    final request = http.MultipartRequest(
      'POST',
      _uri('/reconstruct'),
    );

    request.files.add(
      await http.MultipartFile.fromPath(
        'file',
        imagePath,
        filename: 'furnishar_object.png',
        contentType: _imageMediaType(imagePath),
      ),
    );
    request.fields.addAll({
      'category': 'auto',
      'resolution': resolution.toString(),
      'remove_bg': removeBackground.toString(),
      'segmentation': segmentationEnabled.toString(),
      'texture_res': textureResolution.toString(),
      'include_geometry': 'false',
    });

    final streamed = await request.send().timeout(const Duration(minutes: 10));
    final response = await http.Response.fromStream(streamed);

    if (response.statusCode < 200 || response.statusCode >= 300) {
      throw HttpException(
        'Backend reconstruction failed (${response.statusCode}): ${response.body}',
        uri: _uri('/reconstruct'),
      );
    }

    return ReconstructionResult.fromJson(
      jsonDecode(response.body) as Map<String, dynamic>,
    );
  }
}
