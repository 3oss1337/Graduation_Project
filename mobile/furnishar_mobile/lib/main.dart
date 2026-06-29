import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:model_viewer_plus/model_viewer_plus.dart';
import 'package:url_launcher/url_launcher.dart';

import 'models/reconstruction_result.dart';
import 'models/vision_result.dart';
import 'services/furnishar_api.dart';

class AppTheme {
  static const background = Color(0xFF080A1F);
  static const surface = Color(0xFF11142D);
  static const surfaceAlt = Color(0xFF191C3A);
  static const primary = Color(0xFF7B61FF);
  static const secondary = Color(0xFF00D1FF);
  static const error = Color(0xFFFF6B9A);
  static const errorContainer = Color(0xFF3B1728);
  static const outline = Color(0xFF2B2F55);
  static const text = Color(0xFFF7F7FB);
  static const textMuted = Color(0xFFA8ABC7);

  static ThemeData dark() {
    final scheme = ColorScheme.fromSeed(
      seedColor: primary,
      brightness: Brightness.dark,
      primary: primary,
      secondary: secondary,
      surface: surface,
      error: error,
    );

    return ThemeData(
      colorScheme: scheme,
      scaffoldBackgroundColor: background,
      useMaterial3: true,
      appBarTheme: const AppBarTheme(
        backgroundColor: background,
        foregroundColor: text,
        elevation: 0,
        surfaceTintColor: Colors.transparent,
      ),
      cardTheme: CardThemeData(
        color: surface,
        surfaceTintColor: Colors.transparent,
        elevation: 0,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(8),
          side: const BorderSide(color: outline),
        ),
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: surfaceAlt,
        prefixIconColor: textMuted,
        labelStyle: const TextStyle(color: textMuted),
        hintStyle: const TextStyle(color: textMuted),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(8),
          borderSide: const BorderSide(color: outline),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(8),
          borderSide: const BorderSide(color: primary, width: 1.5),
        ),
        disabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(8),
          borderSide: const BorderSide(color: outline),
        ),
      ),
      filledButtonTheme: FilledButtonThemeData(
        style: FilledButton.styleFrom(
          backgroundColor: primary,
          foregroundColor: Colors.white,
          disabledBackgroundColor: surfaceAlt,
          disabledForegroundColor: textMuted,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
        ),
      ),
      switchTheme: SwitchThemeData(
        thumbColor: WidgetStateProperty.resolveWith(
          (states) =>
              states.contains(WidgetState.selected) ? secondary : textMuted,
        ),
        trackColor: WidgetStateProperty.resolveWith(
          (states) => states.contains(WidgetState.selected)
              ? secondary.withValues(alpha: 0.28)
              : surfaceAlt,
        ),
      ),
      textTheme: Typography.whiteCupertino.apply(
        bodyColor: text,
        displayColor: text,
      ),
      iconTheme: const IconThemeData(color: textMuted),
      dividerColor: outline,
    );
  }
}

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const FurnishArApp());
}

class FurnishArApp extends StatelessWidget {
  const FurnishArApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'FurnishAR',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.dark(),
      home: const CapturePage(),
    );
  }
}

enum PipelineStage {
  idle,
  capture,
  onnx,
  upload,
  reconstruct,
  complete,
}

class CapturePage extends StatefulWidget {
  const CapturePage({super.key});

  @override
  State<CapturePage> createState() => _CapturePageState();
}

class _CapturePageState extends State<CapturePage> {
  final _picker = ImagePicker();
  final _backendController =
      TextEditingController(text: 'http://10.0.2.2:8000');

  XFile? _capturedImage;
  VisionResult? _visionResult;
  ReconstructionResult? _reconstruction;
  PipelineStage _stage = PipelineStage.idle;
  bool _segmentationEnabled = true;
  bool _busy = false;
  String? _error;

  FurnishArApi get _api =>
      FurnishArApi(baseUrl: _backendController.text.trim());

  @override
  void dispose() {
    _backendController.dispose();
    super.dispose();
  }

  Future<void> _pick(ImageSource source) async {
    final image = await _picker.pickImage(
      source: source,
      imageQuality: 95,
      maxWidth: 1600,
    );
    if (image == null) return;
    setState(() {
      _capturedImage = image;
      _visionResult = null;
      _reconstruction = null;
      _stage = PipelineStage.capture;
      _error = null;
    });
  }

  Future<void> _run() async {
    final image = _capturedImage;
    if (image == null || _busy) return;

    setState(() {
      _busy = true;
      _error = null;
      _stage = PipelineStage.onnx;
    });

    try {
      setState(() => _stage = PipelineStage.reconstruct);
      final result = await _api.reconstructWithBackendVision(
        imagePath: image.path,
        resolution: 256,
        textureResolution: 2048,
        removeBackground: true,
        segmentationEnabled: _segmentationEnabled,
      );
      setState(() {
        _visionResult = VisionResult(
          processedImagePath: image.path,
          category: result.category,
          confidence: result.confidence,
          segmentationUsed: _segmentationEnabled,
        );
        _reconstruction = result;
        _stage = PipelineStage.complete;
      });
    } catch (e) {
      setState(() {
        _error = e.toString();
        _stage = PipelineStage.idle;
      });
    } finally {
      if (mounted) {
        setState(() => _busy = false);
      }
    }
  }

  Future<void> _openAr() async {
    final result = _reconstruction;
    if (result == null) return;
    final url = _api.absoluteUrl(result.viewerUrl);
    await launchUrl(Uri.parse(url), mode: LaunchMode.externalApplication);
  }

  @override
  Widget build(BuildContext context) {
    final result = _reconstruction;
    final glbUrl = result == null ? null : _api.absoluteUrl(result.glbUrl);

    return Scaffold(
      appBar: AppBar(
        title: const Text('FurnishAR'),
        centerTitle: false,
        actions: [
          IconButton(
            tooltip: 'Camera',
            onPressed: _busy ? null : () => _pick(ImageSource.camera),
            icon: const Icon(Icons.photo_camera_outlined),
          ),
          IconButton(
            tooltip: 'Gallery',
            onPressed: _busy ? null : () => _pick(ImageSource.gallery),
            icon: const Icon(Icons.photo_library_outlined),
          ),
        ],
      ),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.fromLTRB(16, 8, 16, 24),
          children: [
            _SettingsPanel(
              controller: _backendController,
              segmentationEnabled: _segmentationEnabled,
              onSegmentationChanged: _busy
                  ? null
                  : (value) => setState(() => _segmentationEnabled = value),
            ),
            const SizedBox(height: 12),
            _ImagePanel(image: _capturedImage),
            const SizedBox(height: 12),
            _PipelinePanel(stage: _stage, busy: _busy),
            if (_visionResult case final vision?) ...[
              const SizedBox(height: 12),
              _ClassificationPanel(vision: vision),
            ],
            if (_error case final error?) ...[
              const SizedBox(height: 12),
              _ErrorPanel(message: error),
            ],
            const SizedBox(height: 16),
            FilledButton.icon(
              onPressed: _capturedImage == null || _busy ? null : _run,
              icon: _busy
                  ? const SizedBox.square(
                      dimension: 18,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Icon(Icons.view_in_ar_outlined),
              label: Text(_busy ? 'Processing' : 'Generate 3D Object'),
            ),
            const SizedBox(height: 16),
            _ViewerPanel(
              glbUrl: glbUrl,
              result: result,
              onOpenAr: result == null ? null : _openAr,
            ),
          ],
        ),
      ),
    );
  }
}

class _SettingsPanel extends StatelessWidget {
  const _SettingsPanel({
    required this.controller,
    required this.segmentationEnabled,
    required this.onSegmentationChanged,
  });

  final TextEditingController controller;
  final bool segmentationEnabled;
  final ValueChanged<bool>? onSegmentationChanged;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Backend', style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 10),
            TextField(
              controller: controller,
              decoration: const InputDecoration(
                labelText: 'Server URL',
                hintText: 'http://192.168.1.20:8000',
                prefixIcon: Icon(Icons.dns_outlined),
              ),
              keyboardType: TextInputType.url,
              enabled: onSegmentationChanged != null,
            ),
            const SizedBox(height: 8),
            SwitchListTile(
              contentPadding: EdgeInsets.zero,
              value: segmentationEnabled,
              onChanged: onSegmentationChanged,
              title: const Text('MobileSAM segmentation'),
              subtitle:
                  const Text('Backend pipeline is used for release builds'),
              secondary: const Icon(Icons.auto_fix_high_outlined),
            ),
          ],
        ),
      ),
    );
  }
}

class _ImagePanel extends StatelessWidget {
  const _ImagePanel({required this.image});

  final XFile? image;

  @override
  Widget build(BuildContext context) {
    return AspectRatio(
      aspectRatio: 4 / 3,
      child: DecoratedBox(
        decoration: BoxDecoration(
          color: AppTheme.surfaceAlt,
          borderRadius: BorderRadius.circular(8),
          border: Border.all(color: AppTheme.outline),
        ),
        child: image == null
            ? const Center(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(Icons.add_a_photo_outlined, size: 40),
                    SizedBox(height: 10),
                    Text('Capture or choose a furniture photo'),
                  ],
                ),
              )
            : ClipRRect(
                borderRadius: BorderRadius.circular(8),
                child: Image.file(File(image!.path), fit: BoxFit.cover),
              ),
      ),
    );
  }
}

class _PipelinePanel extends StatelessWidget {
  const _PipelinePanel({required this.stage, required this.busy});

  final PipelineStage stage;
  final bool busy;

  @override
  Widget build(BuildContext context) {
    final steps = [
      (PipelineStage.capture, 'Image'),
      (PipelineStage.onnx, 'ONNX'),
      (PipelineStage.upload, 'Upload'),
      (PipelineStage.reconstruct, '3D'),
      (PipelineStage.complete, 'Done'),
    ];
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Row(
          children: [
            for (final item in steps) ...[
              Expanded(
                child: _StepChip(
                  label: item.$2,
                  active: item.$1 == stage,
                  done: item.$1.index < stage.index,
                ),
              ),
              if (item != steps.last) const SizedBox(width: 6),
            ],
          ],
        ),
      ),
    );
  }
}

class _StepChip extends StatelessWidget {
  const _StepChip({
    required this.label,
    required this.active,
    required this.done,
  });

  final String label;
  final bool active;
  final bool done;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final color = done || active ? scheme.primary : Colors.white24;
    return AnimatedContainer(
      duration: const Duration(milliseconds: 180),
      height: 34,
      alignment: Alignment.center,
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(6),
        border: Border.all(color: color),
        color: active
            ? scheme.primary.withValues(alpha: 0.14)
            : Colors.transparent,
      ),
      child: Text(
        label,
        overflow: TextOverflow.ellipsis,
        style:
            TextStyle(color: color, fontSize: 12, fontWeight: FontWeight.w700),
      ),
    );
  }
}

class _ClassificationPanel extends StatelessWidget {
  const _ClassificationPanel({required this.vision});

  final VisionResult vision;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: ListTile(
        leading: const Icon(Icons.category_outlined),
        title: Text(vision.category.toUpperCase()),
        subtitle: Text(
          '${(vision.confidence * 100).toStringAsFixed(1)}% confidence'
          ' - ${vision.segmentationUsed ? 'segmented' : 'no segmentation'}',
        ),
      ),
    );
  }
}

class _ErrorPanel extends StatelessWidget {
  const _ErrorPanel({required this.message});

  final String message;

  @override
  Widget build(BuildContext context) {
    return Card(
      color: AppTheme.errorContainer,
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Text(message, style: const TextStyle(color: AppTheme.error)),
      ),
    );
  }
}

class _ViewerPanel extends StatelessWidget {
  const _ViewerPanel({
    required this.glbUrl,
    required this.result,
    required this.onOpenAr,
  });

  final String? glbUrl;
  final ReconstructionResult? result;
  final VoidCallback? onOpenAr;

  @override
  Widget build(BuildContext context) {
    final url = glbUrl;
    return Card(
      clipBehavior: Clip.antiAlias,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          SizedBox(
            height: 360,
            child: url == null
                ? const Center(child: Text('Generated GLB will appear here'))
                : ModelViewer(
                    src: url,
                    alt: 'Reconstructed furniture',
                    ar: true,
                    arModes: const ['webxr', 'scene-viewer', 'quick-look'],
                    autoRotate: true,
                    cameraControls: true,
                    backgroundColor: AppTheme.background,
                  ),
          ),
          if (result case final r?)
            Padding(
              padding: const EdgeInsets.fromLTRB(14, 12, 14, 14),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    '${r.category.toUpperCase()} - ${(r.confidence * 100).toStringAsFixed(1)}%',
                    style: Theme.of(context).textTheme.titleMedium,
                  ),
                  const SizedBox(height: 4),
                  Text(
                    '${r.vertexCount} vertices - ${r.faceCount} faces - ${r.timeSeconds.toStringAsFixed(1)}s',
                    style: Theme.of(context).textTheme.bodySmall,
                  ),
                  const SizedBox(height: 12),
                  FilledButton.icon(
                    onPressed: onOpenAr,
                    icon: const Icon(Icons.open_in_new_outlined),
                    label: const Text('Open AR Viewer'),
                  ),
                ],
              ),
            ),
        ],
      ),
    );
  }
}
