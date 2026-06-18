import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:yaru/yaru.dart';

import '../generation/providers/generation_provider.dart';
import '../../features/params/providers/params_provider.dart';
import 'providers/output_provider.dart';

/// State-driven right panel (per D-12, D-13, D-14, and UI-SPEC Right Panel States).
///
/// Renders one of five distinct states:
///   1. idle: icon + instructional text
///   2. generating (pre-progress): indeterminate spinner
///   3. generating (with progress): linear progress bar + step counter
///   4. complete: generated image (BoxFit.contain) + Save button
///   5. error: error icon + message
///
/// The Save button remains visible after saving so the user can save to
/// a different location (UI-SPEC Save Flow point 6). A SnackBar confirms
/// the save path for 4 seconds (D-14).
class OutputPanel extends ConsumerWidget {
  const OutputPanel({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final generationState = ref.watch(generationProvider);
    final colorScheme = Theme.of(context).colorScheme;

    return Container(
      color: colorScheme.surface,
      child: Center(
        child: switch (generationState.status) {
          GenerationStatus.idle => _buildIdleState(context, colorScheme),
          GenerationStatus.generating =>
            _buildGeneratingState(context, generationState),
          GenerationStatus.complete =>
            _buildCompleteState(context, ref, generationState),
          GenerationStatus.error =>
            _buildErrorState(context, generationState, colorScheme),
        },
      ),
    );
  }

  /// Idle state: large icon + instructional text (per D-12).
  Widget _buildIdleState(BuildContext context, ColorScheme colorScheme) {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Icon(
          YaruIcons.image,
          size: 64,
          color: colorScheme.onSurface.withValues(alpha: 0.38),
        ),
        const SizedBox(height: 16),
        Text(
          'Configure parameters and press Generate',
          style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                color: colorScheme.onSurface.withValues(alpha: 0.6),
              ),
        ),
      ],
    );
  }

  /// Generating state: spinner before first progress event, then linear
  /// progress bar + step counter (per D-13, GEN-03, GEN-04).
  Widget _buildGeneratingState(
    BuildContext context,
    GenerationState state,
  ) {
    // Before first progress event (step == 0): show indeterminate spinner.
    if (state.currentStep == 0) {
      return const YaruCircularProgressIndicator();
    }

    // With progress: show linear progress bar + step counter.
    return Padding(
      padding: const EdgeInsets.all(32),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          YaruLinearProgressIndicator(
            value: state.currentStep / state.totalSteps,
          ),
          const SizedBox(height: 16),
          Text(
            'Step ${state.currentStep} / ${state.totalSteps}',
            style: Theme.of(context).textTheme.bodyMedium,
          ),
        ],
      ),
    );
  }

  /// Complete state: generated image + Save button (per OUT-02, OUT-03, OUT-04).
  ///
  /// The image is displayed with [BoxFit.contain] to maintain aspect ratio.
  /// The Save button calls [OutputNotifier.saveImage] and remains visible
  /// after saving so the user can save to another location (UI-SPEC Save
  /// Flow point 6).
  Widget _buildCompleteState(
    BuildContext context,
    WidgetRef ref,
    GenerationState state,
  ) {
    final params = ref.watch(paramsProvider);

    return Padding(
      padding: const EdgeInsets.all(32),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          if (state.imagePath != null)
            Flexible(
              child: Image.file(
                File(state.imagePath!),
                fit: BoxFit.contain,
              ),
            ),
          const SizedBox(height: 16),
          ElevatedButton(
            onPressed: () {
              if (state.imagePath != null) {
                ref.read(outputProvider.notifier).saveImage(
                      state.imagePath!,
                      params.selectedPreset,
                      params.seed,
                      context,
                    );
              }
            },
            child: const Text('Save'),
          ),
        ],
      ),
    );
  }

  /// Error state: error icon + message display.
  Widget _buildErrorState(
    BuildContext context,
    GenerationState state,
    ColorScheme colorScheme,
  ) {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Icon(
          YaruIcons.error,
          size: 64,
          color: colorScheme.error,
        ),
        const SizedBox(height: 16),
        Text(
          state.errorMessage ?? 'An unknown error occurred',
          style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                color: colorScheme.error,
              ),
        ),
      ],
    );
  }
}
