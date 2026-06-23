---
quick_id: 260623-oer
slug: fix-stale-preview-bug-clear-preview-at-g
description: "Fix stale preview bug: clear preview at generation start"
date: 2026-06-23
status: in_progress
---

# Quick Task 260623-oer: Fix Stale Preview Bug

## Problem

Between successive generations, the old preview image is visible when the new
generation's first step fires. Root cause: Flutter widget reconciler reuses the
`Image` element at `Column[0]` when transitioning complete → generating-step-1
(both are `Padding > Column > Flexible(Image)`). With `gaplessPlayback: true`,
the old final image stays rendered while the new preview loads.

## Fix

Add a `generationId` counter to `GenerationState`. Increment on every `generate()`
call. Use `ValueKey(state.generationId)` on `Image.memory` in
`_buildGeneratingState`. Key null (Image.file, complete) → Key(N) (Image.memory,
generating) forces element disposal and fresh render.

## Files to Modify

- `gui/lib/features/generation/providers/generation_provider.dart`
  - Add `final int generationId` field (default 0) to `GenerationState`
  - Add `int _generationId = 0` to `GenerationNotifier`
  - Increment `_generationId` at start of `generate()`, thread through all states
  - Update `copyWith` to include `generationId`

- `gui/lib/features/output/output_panel.dart`
  - Add `key: ValueKey(state.generationId)` to `Image.memory` in
    `_buildGeneratingState`

## Tasks

1. Update `GenerationState` — add `generationId` field + copyWith
2. Update `GenerationNotifier.generate()` — increment and thread `generationId`
3. Update `output_panel.dart` — add `ValueKey` to `Image.memory`
4. Verify: `flutter analyze gui/lib/`
