import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../../shared/models/preset_catalog.dart';
import '../../generation/providers/generation_provider.dart';
import '../providers/params_provider.dart';

/// Model section with Preset and Weights dropdowns (per D-02).
///
/// Weights dropdown is visible but disabled with "N/A" label when the
/// selected preset has no weight variants (per D-06). Both dropdowns
/// disable during generation (per GEN-02).
class ModelSection extends ConsumerWidget {
  const ModelSection({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final params = ref.watch(paramsProvider);
    final generationState = ref.watch(generationProvider);
    final isGenerating =
        generationState.status == GenerationStatus.generating;
    final hasWeights = PresetCatalog.hasWeights(params.selectedPreset);
    final weights = PresetCatalog.getWeights(params.selectedPreset);

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          InputDecorator(
            decoration: const InputDecoration(
              labelText: 'Preset',
              border: OutlineInputBorder(),
            ),
            child: DropdownButtonHideUnderline(
              child: DropdownButton<String>(
                value: params.selectedPreset,
                isExpanded: true,
                isDense: true,
                items: PresetCatalog.presetNames
                    .map(
                      (name) =>
                          DropdownMenuItem(value: name, child: Text(name)),
                    )
                    .toList(),
                onChanged: isGenerating
                    ? null
                    : (value) {
                        if (value != null) {
                          ref
                              .read(paramsProvider.notifier)
                              .setPreset(value);
                        }
                      },
              ),
            ),
          ),
          const SizedBox(height: 12),
          InputDecorator(
            decoration: const InputDecoration(
              labelText: 'Weights',
              border: OutlineInputBorder(),
            ),
            child: DropdownButtonHideUnderline(
              child: DropdownButton<String?>(
                value: hasWeights ? params.selectedWeight : null,
                isExpanded: true,
                isDense: true,
                items: hasWeights
                    ? [
                        const DropdownMenuItem<String?>(
                          value: null,
                          child: Text('Default'),
                        ),
                        ...weights.map(
                          (w) => DropdownMenuItem<String?>(
                            value: w,
                            child: Text(w),
                          ),
                        ),
                      ]
                    : const [
                        DropdownMenuItem<String?>(
                          value: null,
                          child: Text('N/A'),
                        ),
                      ],
                onChanged: (isGenerating || !hasWeights)
                    ? null
                    : (value) {
                        ref.read(paramsProvider.notifier).setWeight(value);
                      },
              ),
            ),
          ),
        ],
      ),
    );
  }
}
