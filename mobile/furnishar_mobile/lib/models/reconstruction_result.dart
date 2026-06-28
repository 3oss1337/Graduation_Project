class ReconstructionResult {
  const ReconstructionResult({
    required this.sessionId,
    required this.glbUrl,
    required this.viewerUrl,
    required this.category,
    required this.confidence,
    required this.vertexCount,
    required this.faceCount,
    required this.timeSeconds,
  });

  final String sessionId;
  final String glbUrl;
  final String viewerUrl;
  final String category;
  final double confidence;
  final int vertexCount;
  final int faceCount;
  final double timeSeconds;

  factory ReconstructionResult.fromJson(Map<String, dynamic> json) {
    return ReconstructionResult(
      sessionId: json['session_id'] as String,
      glbUrl: json['glb_url'] as String,
      viewerUrl: json['viewer_url'] as String? ?? '/viewer/${json['session_id']}',
      category: json['category'] as String? ?? 'object',
      confidence: (json['confidence'] as num? ?? 0).toDouble(),
      vertexCount: (json['n_vertices'] as num? ?? 0).toInt(),
      faceCount: (json['n_faces'] as num? ?? 0).toInt(),
      timeSeconds: (json['time_sec'] as num? ?? 0).toDouble(),
    );
  }
}
