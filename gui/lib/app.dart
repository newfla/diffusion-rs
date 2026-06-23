import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:multi_split_view/multi_split_view.dart';
import 'package:yaru/yaru.dart';

import 'features/generation/providers/generation_provider.dart';
import 'features/output/output_panel.dart';
import 'features/params/params_panel.dart';
import 'features/params/providers/params_provider.dart';
import 'shared/theme/theme_provider.dart';
import 'shared/widgets/drag_handle.dart';

/// Root application widget using Yaru theme (per RESEARCH.md Pitfall 1).
///
/// Wraps the entire app in [YaruTheme] builder pattern to ensure proper
/// theme initialization. The [themeModeProvider] drives Light/System/Dark
/// switching without app restart.
class DiffusionRsApp extends ConsumerWidget {
  const DiffusionRsApp({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final themeMode = ref.watch(themeModeProvider);

    return YaruTheme(
      builder: (context, yaru, child) {
        return MaterialApp(
          title: 'DiffusionRS GUI',
          theme: yaru.theme,
          darkTheme: yaru.darkTheme,
          themeMode: themeMode,
          debugShowCheckedModeBanner: false,
          home: const _MainLayout(),
        );
      },
    );
  }
}

/// Main layout with AppBar (title + theme toggle) and two-panel body.
///
/// Wraps the body in [CallbackShortcuts] + [Focus] for Cmd/Ctrl+Enter
/// keyboard shortcut (per GEN-06, RESEARCH.md Pattern 4, Pitfall 2).
class _MainLayout extends ConsumerStatefulWidget {
  const _MainLayout();

  @override
  ConsumerState<_MainLayout> createState() => _MainLayoutState();
}

class _MainLayoutState extends ConsumerState<_MainLayout> {
  late final MultiSplitViewController _splitController;

  @override
  void initState() {
    super.initState();
    _splitController = MultiSplitViewController(
      areas: [
        Area(flex: 2, min: 320), // Left panel: 40% default, min 320px
        Area(flex: 3, min: 280), // Right panel: 60% default, min 280px
      ],
    );
  }

  @override
  void dispose() {
    _splitController.dispose();
    super.dispose();
  }

  /// Triggers generation if prompt is non-empty and not already generating.
  void _onGenerate() {
    final state = ref.read(generationProvider);
    if (state.status == GenerationStatus.generating) return;

    final params = ref.read(paramsProvider);
    if (params.prompt.trim().isEmpty) return;

    ref.read(generationProvider.notifier).generate(params.toMap());
  }

  @override
  Widget build(BuildContext context) {
    final themeMode = ref.watch(themeModeProvider);

    return Scaffold(
      appBar: AppBar(
        title: Text(
          'DiffusionRS GUI',
          style: Theme.of(context).textTheme.titleLarge,
        ),
        actions: [
          Padding(
            padding: const EdgeInsets.only(right: 16),
            child: SegmentedButton<ThemeMode>(
              segments: const [
                ButtonSegment(
                  value: ThemeMode.light,
                  label: Text('Light'),
                ),
                ButtonSegment(
                  value: ThemeMode.system,
                  label: Text('System'),
                ),
                ButtonSegment(
                  value: ThemeMode.dark,
                  label: Text('Dark'),
                ),
              ],
              selected: {themeMode},
              onSelectionChanged: (selection) {
                ref
                    .read(themeModeProvider.notifier)
                    .setThemeMode(selection.first);
              },
            ),
          ),
        ],
      ),
      // CallbackShortcuts for Cmd/Ctrl+Enter (per GEN-06)
      // Focus with autofocus to maintain focus after divider interaction
      // (per RESEARCH.md Pitfall 2)
      body: CallbackShortcuts(
        bindings: {
          const SingleActivator(LogicalKeyboardKey.enter, meta: true):
              _onGenerate,
          const SingleActivator(LogicalKeyboardKey.enter, control: true):
              _onGenerate,
        },
        child: Focus(
          autofocus: true,
          child: MultiSplitView(
            axis: Axis.horizontal,
            controller: _splitController,
            dividerBuilder:
                (axis, index, resizable, dragging, highlighted, themeData) {
              return DragHandle(
                isDragging: dragging,
                isHighlighted: highlighted,
              );
            },
            builder: (context, area) {
              final index = _splitController.areas.indexOf(area);
              if (index == 0) {
                return const ParamsPanel();
              }
              return const OutputPanel();
            },
          ),
        ),
      ),
    );
  }
}
