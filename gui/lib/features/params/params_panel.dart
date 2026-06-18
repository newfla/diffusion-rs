import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:yaru/yaru.dart';

import '../generation/providers/generation_provider.dart';
import 'providers/params_provider.dart';
import 'sections/advanced_section.dart';
import 'sections/generation_section.dart';
import 'sections/model_section.dart';
import 'sections/postproc_section.dart';

/// Left panel containing the parameter form in 4 collapsible sections
/// and a pinned Generate button at the bottom.
///
/// Uses [YaruExpansionPanel] for the collapsible section layout.
/// Per D-03, Model and Generation are expanded by default;
/// Post-processing and Advanced are collapsed by default.
/// Generate button is pinned outside the scroll area (per UI-SPEC Layout).
class ParamsPanel extends ConsumerWidget {
  const ParamsPanel({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final generationState = ref.watch(generationProvider);
    final params = ref.watch(paramsProvider);
    final isGenerating =
        generationState.status == GenerationStatus.generating;
    final promptEmpty = params.prompt.trim().isEmpty;

    return Column(
      children: [
        Expanded(
          child: SingleChildScrollView(
            padding: const EdgeInsets.only(top: 8),
            child: YaruExpansionPanel(
              // Per D-03: Model + Generation expanded, Post-processing + Advanced collapsed
              isInitiallyExpanded: const [true, true, false, false],
              // Allow multiple sections to be open at the same time
              collapseOnExpand: false,
              placeDividers: true,
              shrinkWrap: true,
              scrollPhysics: const NeverScrollableScrollPhysics(),
              headers: [
                Text(
                  'Model',
                  style: Theme.of(context).textTheme.titleMedium,
                ),
                Text(
                  'Generation',
                  style: Theme.of(context).textTheme.titleMedium,
                ),
                Text(
                  'Post-processing',
                  style: Theme.of(context).textTheme.titleMedium,
                ),
                Text(
                  'Advanced',
                  style: Theme.of(context).textTheme.titleMedium,
                ),
              ],
              children: const [
                ModelSection(),
                GenerationSection(),
                PostprocSection(),
                AdvancedSection(),
              ],
            ),
          ),
        ),
        // Generate button pinned at bottom with 16px padding (per UI-SPEC Layout)
        Padding(
          padding: const EdgeInsets.all(16),
          child: SizedBox(
            width: double.infinity,
            child: ElevatedButton(
              onPressed: (isGenerating || promptEmpty)
                  ? null
                  : () {
                      final paramsMap = ref.read(paramsProvider).toMap();
                      ref
                          .read(generationProvider.notifier)
                          .generate(paramsMap);
                    },
              child: Text(isGenerating ? 'Generating...' : 'Generate'),
            ),
          ),
        ),
      ],
    );
  }
}
