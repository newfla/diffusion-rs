---
quick_id: 260623-p1j
status: complete
completed: 2026-06-23
commit: 09b67f1
---

# Quick Task 260623-p1j: Fix Save Button — Disable When Final Image Not Ready

## Problem

The Rust backend emits multiple `isComplete=true` events — one per generation
phase (text encoder, UNet sampling, VAE decode, etc.). Intermediate complete
events arrive with `previewImage=null`, so `imagePath` stays null in
`GenerationState`. The app enters `GenerationStatus.complete` state but has no
image to save. The Save button was unconditionally rendered with an `onPressed`
callback, making it appear enabled and interactive even with no image behind it.

## Fix

Single-line change in `gui/lib/features/output/output_panel.dart`,
`_buildCompleteState`:

```dart
onPressed: state.imagePath != null
    ? () => ref.read(outputProvider.notifier).saveImage(...)
    : null,
```

`onPressed: null` is the standard Flutter idiom for a disabled button —
the theme renders it visually muted and it ignores taps. The button remains
visible (as a placeholder in the UI layout) but is not interactive until
the final image is written to disk.

Commit: `09b67f1`
