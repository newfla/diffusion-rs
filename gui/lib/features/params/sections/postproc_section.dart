import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../generation/providers/generation_provider.dart';
import '../providers/params_provider.dart';

/// Post-processing section with preview, upscaler, and upscaler scale (per D-02).
///
/// The upscaler scale field is visible only when upscaler is not "None"
/// (per FORM-12). All fields disable during generation (per GEN-02).
class PostprocSection extends ConsumerStatefulWidget {
  const PostprocSection({super.key});

  @override
  ConsumerState<PostprocSection> createState() => _PostprocSectionState();
}

class _PostprocSectionState extends ConsumerState<PostprocSection> {
  late final TextEditingController _scaleController;

  static const _previewModes = ['Fast', 'Accurate'];

  static const _upscalerModes = [
    'None',
    'RealESRGAN_x4plus',
    'RealESRGAN_x4plus_anime_6B',
    'ESRGAN_4x',
    'RealESRGAN_x2plus',
    'RealESRGAN_x4plus_netD',
    'ESRGAN_1x',
    'RealESRGAN_x2_SA',
    'RealESRGAN_x4_Anime',
  ];

  @override
  void initState() {
    super.initState();
    final scale = ref.read(paramsProvider).upscalerScale;
    _scaleController = TextEditingController(text: scale.toString());
  }

  @override
  void dispose() {
    _scaleController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final params = ref.watch(paramsProvider);
    final generationState = ref.watch(generationProvider);
    final isGenerating = generationState.status == GenerationStatus.generating;
    final showScale = params.upscalerMode != 'None';

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Preview dropdown (per FORM-10)
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
          const SizedBox(height: 12),

          // Upscaler dropdown (per FORM-11)
          InputDecorator(
            decoration: const InputDecoration(
              labelText: 'Upscaler',
              border: OutlineInputBorder(),
            ),
            child: DropdownButtonHideUnderline(
              child: DropdownButton<String>(
                value: params.upscalerMode,
                isExpanded: true,
                isDense: true,
                items: _upscalerModes
                    .map((m) => DropdownMenuItem(value: m, child: Text(m)))
                    .toList(),
                onChanged: isGenerating
                    ? null
                    : (value) {
                        if (value != null) {
                          ref
                              .read(paramsProvider.notifier)
                              .setUpscalerMode(value);
                        }
                      },
              ),
            ),
          ),

          // Upscaler scale: visible only when upscaler is not "None" (per FORM-12)
          if (showScale) ...[
            const SizedBox(height: 12),
            TextField(
              controller: _scaleController,
              enabled: !isGenerating,
              decoration: const InputDecoration(
                labelText: 'Scale factor',
                border: OutlineInputBorder(),
              ),
              keyboardType: const TextInputType.numberWithOptions(
                decimal: true,
              ),
              inputFormatters: [
                FilteringTextInputFormatter.allow(RegExp(r'^\d*\.?\d*')),
              ],
              onChanged: (value) {
                final parsed = double.tryParse(value);
                if (parsed != null && parsed > 0) {
                  ref.read(paramsProvider.notifier).setUpscalerScale(parsed);
                }
              },
            ),
          ],
        ],
      ),
    );
  }
}
