import '../../../shared/models/progress_event.dart';

/// Abstract interface for image generation.
///
/// This is the Phase 1 / Phase 2 seam (per D-08).
/// Phase 1: [MockGenerationService] implements this with simulated progress.
/// Phase 2: RustGenerationService will replace it via a single provider
/// line change -- no structural refactor needed.
abstract class GenerationService {
  /// Starts an image generation and returns a stream of progress events.
  ///
  /// The [params] map contains the generation parameters collected from the
  /// form fields. The stream completes when generation finishes.
  Stream<ProgressEvent> generate(Map<String, dynamic> params);
}
