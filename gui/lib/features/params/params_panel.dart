import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../generation/providers/generation_provider.dart';

/// Skeleton left panel containing the parameter form and Generate button.
///
/// Detailed form fields (preset, prompt, steps, etc.) come in Plan 02.
/// This version provides the Generate button (per GEN-01) and panel structure.
class ParamsPanel extends ConsumerWidget {
  const ParamsPanel({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final generationState = ref.watch(generationProvider);
    final isGenerating =
        generationState.status == GenerationStatus.generating;

    return Column(
      children: [
        Expanded(
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(16),
            child: Center(
              child: Text(
                'Parameters (coming in next plan)',
                style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                      color: Theme.of(context)
                          .colorScheme
                          .onSurface
                          .withValues(alpha: 0.6),
                    ),
              ),
            ),
          ),
        ),
        Padding(
          padding: const EdgeInsets.all(16),
          child: SizedBox(
            width: double.infinity,
            child: ElevatedButton(
              onPressed: isGenerating
                  ? null
                  : () {
                      ref
                          .read(generationProvider.notifier)
                          .generate({});
                    },
              child: Text(isGenerating ? 'Generating...' : 'Generate'),
            ),
          ),
        ),
      ],
    );
  }
}
