import 'dart:typed_data';

import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../../shared/models/progress_event.dart';
import '../../../shared/services/temp_directory_manager.dart';
import '../../../src/rust/api.dart';
import '../../../src/rust/gui_params.dart';
import 'generation_service.dart';

/// Real implementation of [GenerationService] that calls diffusion-rs
/// via flutter_rust_bridge FFI bindings (FRB-09).
///
/// Converts the params Map to [GuiParams], calls [generateImageStream],
/// and yields [ProgressEvent] instances from the Rust stream. Preview
/// image bytes (read from disk by the Rust relay thread per D-03) are
/// forwarded as [ProgressEvent.previewImage]. The final image bytes
/// arrive via [GuiProgressEvent.finalImage] on the completion event.
class RustGenerationService implements GenerationService {
  final Ref _ref;

  RustGenerationService(this._ref);

  @override
  Stream<ProgressEvent> generate(Map<String, dynamic> params) async* {
    final tempManager = _ref.read(tempDirectoryManagerProvider);
    final sessionPath = tempManager.sessionPath;
    final timestamp = DateTime.now().millisecondsSinceEpoch;
    final previewPath = '$sessionPath/preview_$timestamp.png';
    final outputPath = '$sessionPath/output_$timestamp.png';

    // Map "None" string values to null for optional Rust fields.
    String? nullIfNone(String? value) =>
        (value == null || value == 'None') ? null : value;

    // Convert the params Map to FRB GuiParams DTO.
    final guiParams = GuiParams(
      preset: params['preset'] as String,
      weight: nullIfNone(params['weight'] as String?),
      prompt: params['prompt'] as String,
      negativePrompt: _nonEmptyOrNull(params['negativePrompt'] as String?),
      steps: params['steps'] as int?,
      width: params['width'] as int?,
      height: params['height'] as int?,
      batchCount: 1,
      seed: params['seed'] as int? ?? -1,
      cacheMode: nullIfNone(params['cacheMode'] as String?),
      previewMode: params['previewMode'] as String? ?? 'Fast',
      upscaler: nullIfNone(params['upscalerMode'] as String?),
      upscalerScale: (params['upscalerScale'] as num?)?.toDouble() ?? 2.0,
      token: _nonEmptyOrNull(params['token'] as String?),
      lowVram: params['lowVram'] as bool? ?? false,
      previewOutput: previewPath,
      output: outputPath,
    );

    // Call FRB-generated function that returns Stream<GuiProgressEvent>.
    await for (final event in generateImageStream(params: guiParams)) {
      // Final image event (finalImage populated): yield as complete.
      if (event.finalImage != null) {
        yield ProgressEvent(
          step: event.steps > 0 ? event.steps : event.step,
          steps: event.steps > 0 ? event.steps : event.step,
          time: event.time,
          previewImage: event.finalImage,
        );
      } else {
        // Progress event: forward preview image bytes if available.
        yield ProgressEvent(
          step: event.step,
          steps: event.steps,
          time: event.time,
          previewImage: event.previewImage != null
              ? Uint8List.fromList(event.previewImage!)
              : null,
        );
      }
    }
  }

  /// Returns null if the string is null or empty.
  String? _nonEmptyOrNull(String? value) =>
      (value == null || value.isEmpty) ? null : value;
}
