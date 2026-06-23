import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:yaru/yaru.dart';

import '../../shared/widgets/error_dialog.dart';
import '../generation/providers/generation_provider.dart';
import '../../features/params/providers/params_provider.dart';
import 'providers/output_provider.dart';

/// State-driven right panel (per D-12, D-13, D-14, and UI-SPEC Right Panel States).
///
/// Renders one of five distinct states:
///   1. idle: icon + instructional text
///   2. generating (pre-progress): spinner + "Downloading model..." (D-04)
///   3. generating (with progress): live preview image + progress bar (D-01/D-02)
///   4. complete: generated image (BoxFit.contain) + Save button
///   5. error: error icon + message + modal error dialog (D-05)
///
/// The Save button remains visible after saving so the user can save to
/// a different location (UI-SPEC Save Flow point 6). A SnackBar confirms
/// the save path for 4 seconds (D-14).
class OutputPanel extends ConsumerStatefulWidget {
  const OutputPanel({super.key});

  @override
  ConsumerState<OutputPanel> createState() => _OutputPanelState();
}

class _OutputPanelState extends ConsumerState<OutputPanel> {
  @override
  void initState() {
    super.initState();
    // Listen for error state transitions to trigger the modal error dialog.
    // Using a post-frame callback ensures the dialog shows after the widget
    // tree has finished building (per D-05).
    ref.listenManual<GenerationState>(generationProvider, (previous, next) {
      if (next.status == GenerationStatus.error &&
          (previous == null ||
              previous.status != GenerationStatus.error) &&
          next.errorMessage != null &&
          mounted) {
        WidgetsBinding.instance.addPostFrameCallback((_) {
          if (mounted) {
            showErrorDialog(context, next.errorMessage!);
          }
        });
      }
    });
  }

  @override
  Widget build(BuildContext context) {
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

  /// Generating state: "Downloading model..." spinner when no progress yet,
  /// then live preview image + progress bar once inference starts (D-04, D-01).
  Widget _buildGeneratingState(
    BuildContext context,
    GenerationState state,
  ) {
    // Before first progress event (step == 0): show spinner + "Downloading
    // model..." text (per D-04). This covers the model download phase before
    // inference starts.
    if (state.currentStep == 0) {
      return Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const YaruCircularProgressIndicator(),
          const SizedBox(height: 16),
          Text(
            'Downloading model...',
            style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                  color: Theme.of(context)
                      .colorScheme
                      .onSurface
                      .withValues(alpha: 0.6),
                ),
          ),
        ],
      );
    }

    // With progress: show live preview image (if available) above progress bar.
    return Padding(
      padding: const EdgeInsets.all(32),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          // Live preview image display (D-01, D-02, D-03).
          // If previewBytes is available, show the intermediate frame.
          // If null for this step, the column simply omits the image
          // (graceful degradation per D-03).
          if (state.previewBytes != null)
            Flexible(
              key: ValueKey(state.generationId),
              child: Image.memory(
                state.previewBytes!,
                fit: BoxFit.contain,
              ),
            ),
          if (state.previewBytes != null) const SizedBox(height: 16),
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
  /// The modal error dialog is triggered separately via the listener in
  /// [initState] (per D-05). This inline display serves as a fallback.
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
