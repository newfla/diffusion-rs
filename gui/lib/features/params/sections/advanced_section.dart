import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../generation/providers/generation_provider.dart';
import '../providers/params_provider.dart';

/// Advanced section: cache, FORM-15 warning, upscaler, upscaler scale, token, low VRAM.
///
/// Upscaler moved here from Post-processing per user feedback.
/// FORM-15 warning shown when upscaler is active and cache is "None" (per D-05).
/// All fields disable during generation (per GEN-02).
class AdvancedSection extends ConsumerStatefulWidget {
  const AdvancedSection({super.key});

  @override
  ConsumerState<AdvancedSection> createState() => _AdvancedSectionState();
}

class _AdvancedSectionState extends ConsumerState<AdvancedSection> {
  late final TextEditingController _scaleController;

  static const _cacheModes = [
    'None',
    'UCACHE',
    'EASYCACHE',
    'DBCACHE',
    'TAYLORSEER',
    'CACHEDIT',
    'SPECTRUM',
  ];

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
    // Also treat intermediate complete events (imagePath null) as "generating"
    // — the Rust API emits isComplete for each phase, not just the final one.
    final isGenerating =
        generationState.status == GenerationStatus.generating ||
        (generationState.status == GenerationStatus.complete &&
            generationState.imagePath == null);
    final showScale = params.upscalerMode != 'None';
    final showUpscalerWarning =
        params.upscalerMode != 'None' && params.cacheMode == 'None';

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Cache dropdown (per FORM-09)
          InputDecorator(
            decoration: const InputDecoration(
              labelText: 'Cache mode',
              border: OutlineInputBorder(),
            ),
            child: DropdownButtonHideUnderline(
              child: DropdownButton<String>(
                value: params.cacheMode,
                isExpanded: true,
                isDense: true,
                items: _cacheModes
                    .map((m) => DropdownMenuItem(value: m, child: Text(m)))
                    .toList(),
                onChanged: isGenerating
                    ? null
                    : (value) {
                        if (value != null) {
                          ref.read(paramsProvider.notifier).setCacheMode(value);
                        }
                      },
              ),
            ),
          ),

          // FORM-15 warning (per D-05)
          if (showUpscalerWarning) ...[
            const SizedBox(height: 8),
            Text(
              'Upscaler is active without caching. Select a cache mode '
              'to avoid recomputing all steps during upscaling.',
              style: Theme.of(context).textTheme.labelMedium?.copyWith(
                color: Theme.of(context).colorScheme.error,
              ),
            ),
          ],
          const SizedBox(height: 12),

          // Upscaler dropdown (moved from Post-processing per user feedback)
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
          const SizedBox(height: 12),

          // Token field with obscureText toggle (per FORM-13)
          TextField(
            enabled: !isGenerating,
            obscureText: !params.tokenVisible,
            decoration: InputDecoration(
              labelText: 'HuggingFace Token',
              hintText: 'Default',
              border: const OutlineInputBorder(),
              suffixIcon: IconButton(
                onPressed: isGenerating
                    ? null
                    : () {
                        ref
                            .read(paramsProvider.notifier)
                            .setTokenVisible(!params.tokenVisible);
                      },
                icon: Icon(
                  params.tokenVisible ? Icons.visibility_off : Icons.visibility,
                ),
                tooltip: params.tokenVisible ? 'Hide token' : 'Show token',
              ),
            ),
            onChanged: (value) {
              ref.read(paramsProvider.notifier).setToken(value);
            },
          ),
          const SizedBox(height: 12),

          // Low VRAM toggle (per FORM-14)
          SwitchListTile(
            title: const Text('Low VRAM mode'),
            value: params.lowVram,
            contentPadding: EdgeInsets.zero,
            onChanged: isGenerating
                ? null
                : (value) {
                    ref.read(paramsProvider.notifier).setLowVram(value);
                  },
          ),
        ],
      ),
    );
  }
}
