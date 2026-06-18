import 'dart:async';
import 'dart:io';

import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../../shared/services/temp_directory_manager.dart';
import '../services/generation_service.dart';
import '../services/mock_generation_service.dart';

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

  const GenerationState({
    this.status = GenerationStatus.idle,
    this.currentStep = 0,
    this.totalSteps = 0,
    this.imagePath,
    this.errorMessage,
  });

  GenerationState copyWith({
    GenerationStatus? status,
    int? currentStep,
    int? totalSteps,
    String? imagePath,
    String? errorMessage,
  }) {
    return GenerationState(
      status: status ?? this.status,
      currentStep: currentStep ?? this.currentStep,
      totalSteps: totalSteps ?? this.totalSteps,
      imagePath: imagePath ?? this.imagePath,
      errorMessage: errorMessage ?? this.errorMessage,
    );
  }
}

/// Riverpod Notifier managing the generation lifecycle state machine.
///
/// The [generate] method transitions through:
///   idle -> generating -> complete (or error)
///
/// On completion, copies the bundled placeholder.png asset to the session
/// temp directory (via [TempDirectoryManager]) and sets
/// [GenerationState.imagePath] so the output panel can display it.
class GenerationNotifier extends Notifier<GenerationState> {
  StreamSubscription? _subscription;

  @override
  GenerationState build() {
    ref.onDispose(() {
      _subscription?.cancel();
    });
    return const GenerationState();
  }

  /// Starts a mock generation run with the given [params].
  Future<void> generate(Map<String, dynamic> params) async {
    // Prevent concurrent generations.
    if (state.status == GenerationStatus.generating) return;

    state = const GenerationState(status: GenerationStatus.generating);

    final service = ref.read(generationServiceProvider);

    try {
      await for (final event in service.generate(params)) {
        if (event.isComplete) {
          // Copy bundled placeholder to session temp directory for display.
          final tempManager = ref.read(tempDirectoryManagerProvider);
          final timestamp = DateTime.now().millisecondsSinceEpoch;
          final outputFile = File(
            '${tempManager.sessionPath}/output_$timestamp.png',
          );

          final byteData = await rootBundle.load('assets/placeholder.png');
          await outputFile.writeAsBytes(
            byteData.buffer.asUint8List(
              byteData.offsetInBytes,
              byteData.lengthInBytes,
            ),
          );

          state = GenerationState(
            status: GenerationStatus.complete,
            currentStep: event.step,
            totalSteps: event.steps,
            imagePath: outputFile.path,
          );
        } else {
          state = GenerationState(
            status: GenerationStatus.generating,
            currentStep: event.step,
            totalSteps: event.steps,
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
/// Phase 1: returns [MockGenerationService].
/// Phase 2: swap this single line to return RustGenerationService.
final generationServiceProvider = Provider<GenerationService>((ref) {
  return MockGenerationService();
});
