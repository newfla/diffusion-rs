import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../generation/providers/generation_provider.dart';
import '../providers/params_provider.dart';

/// Advanced section with cache, FORM-15 warning, token, and low VRAM (per D-02, D-05).
///
/// FORM-15 warning is shown when upscaler is active and cache is "None".
/// Token field stores obscureText toggle state in paramsProvider (not local
/// state) to survive section rebuilds (per RESEARCH.md Pitfall 7).
/// All fields disable during generation (per GEN-02).
class AdvancedSection extends ConsumerWidget {
  const AdvancedSection({super.key});

  static const _cacheModes = [
    'None',
    'UCACHE',
    'EASYCACHE',
    'DBCACHE',
    'TAYLORSEER',
    'CACHEDIT',
    'SPECTRUM',
  ];

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final params = ref.watch(paramsProvider);
    final generationState = ref.watch(generationProvider);
    final isGenerating =
        generationState.status == GenerationStatus.generating;

    // FORM-15 warning condition: upscaler active AND cache is None (per D-05)
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
                    .map(
                      (m) => DropdownMenuItem(value: m, child: Text(m)),
                    )
                    .toList(),
                onChanged: isGenerating
                    ? null
                    : (value) {
                        if (value != null) {
                          ref
                              .read(paramsProvider.notifier)
                              .setCacheMode(value);
                        }
                      },
              ),
            ),
          ),

          // FORM-15 warning text (per D-05, UI-SPEC Copywriting)
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

          // Token field with obscureText toggle (per FORM-13, T-01-05)
          TextField(
            enabled: !isGenerating,
            obscureText: !params.tokenVisible,
            decoration: InputDecoration(
              labelText: 'HuggingFace Token',
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
                  params.tokenVisible
                      ? Icons.visibility_off
                      : Icons.visibility,
                ),
                tooltip:
                    params.tokenVisible ? 'Hide token' : 'Show token',
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
