import 'package:flutter/material.dart';

/// Vertical drag divider for [MultiSplitView].
///
/// Renders an 8px visible handle with a 20px logical hit target (per UI-SPEC).
/// Shows a subtle grip indicator (three horizontal lines) and highlights
/// on hover/drag to provide visual feedback.
class DragHandle extends StatelessWidget {
  final bool isDragging;
  final bool isHighlighted;

  const DragHandle({
    super.key,
    this.isDragging = false,
    this.isHighlighted = false,
  });

  @override
  Widget build(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;
    final isActive = isDragging || isHighlighted;

    return SizedBox(
      width: 20,
      child: Center(
        child: Container(
          width: 8,
          decoration: BoxDecoration(
            color: isActive
                ? colorScheme.surfaceContainerLow
                : Colors.transparent,
            border: Border.symmetric(
              vertical: BorderSide(
                color: colorScheme.outlineVariant,
                width: 0.5,
              ),
            ),
          ),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: List.generate(
              3,
              (index) => Padding(
                padding: const EdgeInsets.symmetric(vertical: 1.5),
                child: Container(
                  width: 4,
                  height: 1,
                  color: isActive
                      ? colorScheme.onSurface.withValues(alpha: 0.6)
                      : colorScheme.onSurface.withValues(alpha: 0.3),
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}
