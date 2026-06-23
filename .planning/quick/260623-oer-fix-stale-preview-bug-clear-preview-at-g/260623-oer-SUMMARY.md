---
quick_id: 260623-oer
status: complete
completed: 2026-06-23
commit: fa04541
---

# Quick Task 260623-oer: Fix Stale Preview Bug

## Result

Fixed the stale preview image appearing between successive generations.

**Root cause:** Flutter's widget reconciler reuses the `Image` element when
transitioning from `_buildCompleteState` (`Image.file`, no key) to
`_buildGeneratingState` (`Image.memory`). Both sit at `Padding > Column >
Flexible[0]` — same runtimeType (`Image`), no key → element reused.
`gaplessPlayback: true` then keeps the old final frame visible while the new
preview loads.

**Fix:** Added `generationId` (monotonic int, default 0) to `GenerationState`.
Incremented via `_generationId++` in `GenerationNotifier.generate()` before the
first state transition. Threaded through all state constructors in `generate()`.
Added `key: ValueKey(state.generationId)` to `Image.memory` in
`_buildGeneratingState`.

**Behaviour:**
- Null key (`Image.file`, complete) → `ValueKey(N)` (new generation) → Flutter
  disposes old element, renders blank until first preview arrives. No stale frame.
- `ValueKey(N)` stable across steps within one generation → `gaplessPlayback`
  still suppresses step-by-step flicker.

**Files changed:**
1. `gui/lib/features/generation/providers/generation_provider.dart` —
   `generationId` field, copyWith param, `_generationId` counter, threading
2. `gui/lib/features/output/output_panel.dart` — `key: ValueKey(state.generationId)`

`flutter analyze lib/` — 1 pre-existing info in generated FRB file (unrelated).

Commit: `fa04541`
