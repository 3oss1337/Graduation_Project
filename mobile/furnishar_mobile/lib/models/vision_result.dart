class VisionResult {
  const VisionResult({
    required this.processedImagePath,
    required this.category,
    required this.confidence,
    required this.segmentationUsed,
  });

  final String processedImagePath;
  final String category;
  final double confidence;
  final bool segmentationUsed;

  factory VisionResult.fromMap(Map<dynamic, dynamic> map) {
    return VisionResult(
      processedImagePath: map['processedImagePath'] as String,
      category: map['category'] as String,
      confidence: (map['confidence'] as num).toDouble(),
      segmentationUsed: map['segmentationUsed'] as bool? ?? false,
    );
  }
}
