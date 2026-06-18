import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../features/params/providers/params_provider.dart';

/// Numeric seed input with a dice button that resets value to -1 (per FORM-08).
///
/// The dice IconButton sets seed to -1 in paramsProvider, meaning "random"
/// (same semantics as the CLI). Tooltip: "Randomize seed" per UI-SPEC.
class SeedField extends ConsumerStatefulWidget {
  const SeedField({super.key, this.enabled = true});

  final bool enabled;

  @override
  ConsumerState<SeedField> createState() => _SeedFieldState();
}

class _SeedFieldState extends ConsumerState<SeedField> {
  late final TextEditingController _controller;

  @override
  void initState() {
    super.initState();
    final seed = ref.read(paramsProvider).seed;
    _controller = TextEditingController(text: seed.toString());
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    // Listen for external seed changes (e.g. dice button reset)
    ref.listen(paramsProvider.select((p) => p.seed), (previous, next) {
      if (_controller.text != next.toString()) {
        _controller.text = next.toString();
      }
    });

    return Row(
      children: [
        Expanded(
          child: TextField(
            controller: _controller,
            enabled: widget.enabled,
            decoration: const InputDecoration(
              labelText: 'Seed',
              border: OutlineInputBorder(),
            ),
            keyboardType: const TextInputType.numberWithOptions(
              signed: true,
            ),
            inputFormatters: [
              FilteringTextInputFormatter.allow(RegExp(r'^-?\d*')),
            ],
            onChanged: (value) {
              final parsed = int.tryParse(value);
              if (parsed != null) {
                ref.read(paramsProvider.notifier).setSeed(parsed);
              }
            },
          ),
        ),
        const SizedBox(width: 8),
        IconButton(
          onPressed: widget.enabled
              ? () {
                  ref.read(paramsProvider.notifier).setSeed(-1);
                }
              : null,
          icon: const Icon(Icons.casino),
          tooltip: 'Randomize seed',
        ),
      ],
    );
  }
}
