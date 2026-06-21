import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../../shared/services/temp_directory_manager.dart';
import '../services/generation_service.dart';
import '../services/rust_generation_service.dart';

/// Generation lifecycle status enum (idle/generating/complete/error).
enum GenerationStatus { idle, generating, complete, error }

/// Immutable state class for the generation lifecycle.
class GenerationState {
  final GenerationStatus status;
  final int currentStep;
  final int totalSteps;

  /// Path to the generated image file on disk. Non-null when status == complete.
  final String? imagePath;

  /// Error message when status == error.
  final String? errorMessage;

  /// Live preview image bytes from the Rust backend (D-01, D-02, D-03).
  /// Populated during generation when the backend produces intermediate frames.
  /// Null when no preview is available yet for the current step.
  final Uint8List? previewBytes;

  const GenerationState({
    this.status = GenerationStatus.idle,
    this.currentStep = 0,
    this.totalSteps = 0,
    this.imagePath,
    this.errorMessage,
    this.previewBytes,
  });

  GenerationState copyWith({
    GenerationStatus? status,
    int? currentStep,
    int? totalSteps,
    String? imagePath,
    String? errorMessage,
    Uint8List? Function()? previewBytesFn,
  }) {
    return GenerationState(
      status: status ?? this.status,
      currentStep: currentStep ?? this.currentStep,
      totalSteps: totalSteps ?? this.totalSteps,
      imagePath: imagePath ?? this.imagePath,
      errorMessage: errorMessage ?? this.errorMessage,
      previewBytes:
          previewBytesFn != null ? previewBytesFn() : previewBytes,
    );
  }
}

/// Riverpod Notifier managing the generation lifecycle state machine.
///
/// The [generate] method transitions through:
///   idle -> generating -> complete (or error)
///
/// On completion, writes the final image bytes from the Rust backend to
/// the session temp directory and sets [GenerationState.imagePath] so the
/// output panel can display it. During generation, preview image bytes
/// are passed through [GenerationState.previewBytes] for live display.
class GenerationNotifier extends Notifier<GenerationState> {
  StreamSubscription? _subscription;

  @override
  GenerationState build() {
    ref.onDispose(() {
      _subscription?.cancel();
    });
    return const GenerationState();
  }

  /// Starts an image generation run with the given [params].
  Future<void> generate(Map<String, dynamic> params) async {
    // Prevent concurrent generations.
    if (state.status == GenerationStatus.generating) return;

    state = const GenerationState(status: GenerationStatus.generating);

    final service = ref.read(generationServiceProvider);

    try {
      await for (final event in service.generate(params)) {
        if (event.isComplete) {
          // Write final image bytes to the session temp directory.
          final tempManager = ref.read(tempDirectoryManagerProvider);
          final timestamp = DateTime.now().millisecondsSinceEpoch;
          final outputFile = File(
            '${tempManager.sessionPath}/output_$timestamp.png',
          );

          if (event.previewImage != null) {
            // Final image bytes arrived from Rust -- write to output file.
            await outputFile.writeAsBytes(event.previewImage!);
          }
          // Fallback: if previewImage is null on completion, check if the
          // Rust backend already wrote the output file directly on disk.
          final fileExists = await outputFile.exists();

          state = GenerationState(
            status: GenerationStatus.complete,
            currentStep: event.step,
            totalSteps: event.steps,
            imagePath: fileExists ? outputFile.path : null,
          );
        } else {
          // In-progress event: pass preview bytes for live display (D-01/D-02).
          state = GenerationState(
            status: GenerationStatus.generating,
            currentStep: event.step,
            totalSteps: event.steps,
            previewBytes: event.previewImage,
          );
        }
      }
    } catch (e) {
      state = GenerationState(
        status: GenerationStatus.error,
        errorMessage: e.toString(),
      );
    }
  }
}

/// Provider for the generation lifecycle state.
final generationProvider =
    NotifierProvider<GenerationNotifier, GenerationState>(
  GenerationNotifier.new,
);

/// Provider for the [GenerationService] implementation.
/// Phase 2: RustGenerationService replaces MockGenerationService (FRB-09).
final generationServiceProvider = Provider<GenerationService>((ref) {
  return RustGenerationService(ref);
});
