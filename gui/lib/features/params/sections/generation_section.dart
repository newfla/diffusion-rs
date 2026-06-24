import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../../shared/widgets/seed_field.dart';
import '../../generation/providers/generation_provider.dart';
import '../providers/params_provider.dart';

/// Generation section: prompt → negative → steps → width/height → seed → preview.
///
/// Preview moved here from Post-processing per user feedback.
/// Steps, width, height show hint "Default" — backend uses model defaults when null.
/// All fields disable when generation is running (per GEN-02).
class GenerationSection extends ConsumerStatefulWidget {
  const GenerationSection({super.key});

  @override
  ConsumerState<GenerationSection> createState() => _GenerationSectionState();
}

class _GenerationSectionState extends ConsumerState<GenerationSection> {
  late final TextEditingController _promptController;
  late final TextEditingController _negativePromptController;
  late final TextEditingController _stepsController;
  late final TextEditingController _widthController;
  late final TextEditingController _heightController;

  static const _previewModes = ['Fast', 'Accurate'];

  @override
  void initState() {
    super.initState();
    final params = ref.read(paramsProvider);
    _promptController = TextEditingController(text: params.prompt);
    _negativePromptController = TextEditingController(
      text: params.negativePrompt,
    );
    _stepsController = TextEditingController(
      text: params.steps?.toString() ?? '',
    );
    _widthController = TextEditingController(
      text: params.width?.toString() ?? '',
    );
    _heightController = TextEditingController(
      text: params.height?.toString() ?? '',
    );

    // Sync controllers when preset changes (preset drives steps/width/height defaults).
    ref.listenManual(paramsProvider.select((p) => p.selectedPreset), (_, _) {
      final p = ref.read(paramsProvider);
      _stepsController.text = p.steps?.toString() ?? '';
      _widthController.text = p.width?.toString() ?? '';
      _heightController.text = p.height?.toString() ?? '';
    });
  }

  @override
  void dispose() {
    _promptController.dispose();
    _negativePromptController.dispose();
    _stepsController.dispose();
    _widthController.dispose();
    _heightController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final params = ref.watch(paramsProvider);
    final generationState = ref.watch(generationProvider);
    // Also treat intermediate complete events (imagePath null) as "generating"
    // — the Rust API emits isComplete for each phase, not just the final one.
    final isGenerating =
        generationState.status == GenerationStatus.generating ||
        (generationState.status == GenerationStatus.complete &&
            generationState.imagePath == null);

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Prompt: required (per FORM-03)
          TextField(
            controller: _promptController,
            enabled: !isGenerating,
            decoration: const InputDecoration(
              labelText: 'Prompt',
              border: OutlineInputBorder(),
              alignLabelWithHint: true,
            ),
            maxLines: null,
            minLines: 3,
            onChanged: (value) {
              ref.read(paramsProvider.notifier).setPrompt(value);
            },
          ),
          const SizedBox(height: 12),

          // Negative prompt: optional, hint "Default" (per FORM-04)
          TextField(
            controller: _negativePromptController,
            enabled: !isGenerating,
            decoration: const InputDecoration(
              labelText: 'Negative prompt',
              hintText: 'Default',
              border: OutlineInputBorder(),
            ),
            onChanged: (value) {
              ref.read(paramsProvider.notifier).setNegativePrompt(value);
            },
          ),
          const SizedBox(height: 12),

          // Steps: preset default auto-filled; empty means backend default
          TextField(
            controller: _stepsController,
            enabled: !isGenerating,
            decoration: const InputDecoration(
              labelText: 'Steps',
              border: OutlineInputBorder(),
            ),
            keyboardType: TextInputType.number,
            inputFormatters: [FilteringTextInputFormatter.digitsOnly],
            onChanged: (value) {
              ref.read(paramsProvider.notifier).setSteps(int.tryParse(value));
            },
          ),
          const SizedBox(height: 12),

          // Width / Height: optional, hint "Default" (per FORM-06)
          Row(
            children: [
              Expanded(
                child: TextField(
                  controller: _widthController,
                  enabled: !isGenerating,
                  decoration: const InputDecoration(
                    labelText: 'Width',
                    border: OutlineInputBorder(),
                  ),
                  keyboardType: TextInputType.number,
                  inputFormatters: [FilteringTextInputFormatter.digitsOnly],
                  onChanged: (value) {
                    ref
                        .read(paramsProvider.notifier)
                        .setWidth(int.tryParse(value));
                  },
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: TextField(
                  controller: _heightController,
                  enabled: !isGenerating,
                  decoration: const InputDecoration(
                    labelText: 'Height',
                    border: OutlineInputBorder(),
                  ),
                  keyboardType: TextInputType.number,
                  inputFormatters: [FilteringTextInputFormatter.digitsOnly],
                  onChanged: (value) {
                    ref
                        .read(paramsProvider.notifier)
                        .setHeight(int.tryParse(value));
                  },
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),

          // Seed with dice button (per FORM-08)
          SeedField(enabled: !isGenerating),
          const SizedBox(height: 12),

          // Preview dropdown (moved from Post-processing per user feedback)
          InputDecorator(
            decoration: const InputDecoration(
              labelText: 'Preview',
              border: OutlineInputBorder(),
            ),
            child: DropdownButtonHideUnderline(
              child: DropdownButton<String>(
                value: params.previewMode,
                isExpanded: true,
                isDense: true,
                items: _previewModes
                    .map((m) => DropdownMenuItem(value: m, child: Text(m)))
                    .toList(),
                onChanged: isGenerating
                    ? null
                    : (value) {
                        if (value != null) {
                          ref
                              .read(paramsProvider.notifier)
                              .setPreviewMode(value);
                        }
                      },
              ),
            ),
          ),
        ],
      ),
    );
  }
}
