import 'dart:typed_data';

/// Mirrors the Rust Progress struct shape (src/api.rs fields step, steps, time).
/// Used by GenerationService to report generation progress back to the UI.
class ProgressEvent {
  final int step;
  final int steps;
  final double time;

  /// Preview image bytes, available when the backend produces intermediate frames.
  /// Null during mock generation (Phase 1).
  final Uint8List? previewImage;

  const ProgressEvent({
    required this.step,
    required this.steps,
    required this.time,
    this.previewImage,
  });

  /// Returns true when the generation has completed all steps.
  bool get isComplete => step >= steps;
}
