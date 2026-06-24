---
quick_id: 260623-omk
status: complete
completed: 2026-06-23
commit: 6bf4f4c
---

# Quick Task 260623-omk: Fix Stale Preview (corrected)

## Problem

Previous fix (260623-oer) placed `key: ValueKey(state.generationId)` BEFORE
the positional `bytes` argument in `Image.memory(...)`. In Dart, named args
cannot precede positional args — this is a syntax error. The hot-reload
compiler rejected the build silently; the app kept running with the pre-fix
code, so the bug appeared unchanged to the user.

## Fix

Two-part correction in `gui/lib/features/output/output_panel.dart`:

1. **Moved `ValueKey` to the `Flexible` wrapper** (not `Image.memory`). This
   is the correct level: when `generationId` changes between generations,
   `canUpdate(Flexible(key='old'), Flexible(key='new'))` returns false →
   Flutter disposes the entire `Flexible` subtree (including `Image`) and
   creates a fresh one. No stale image can persist.

2. **Removed `gaplessPlayback: true`**. Belt-and-suspenders: without it,
   even if element reuse somehow occurred, the `Image` widget would reset to
   blank rather than keeping the old frame. Directly honours the user's
   explicit request to "cancel any reference to the previous preview when
   clicking Generate".

## Why It Works Now

- Between generations (`generationId` 1 → 2): `Flexible(key=1)` → `Flexible(key=2)`
  → `canUpdate = false` → dispose old subtree, create fresh Image widget. ✓
- Within a generation (step 1 → 2 → 3): `Flexible(key=2)` → `Flexible(key=2)`
  → element reused → `Image.memory` updates with new bytes (brief decode
  latency, no old image retained since gaplessPlayback removed). ✓
- On Generate click with step=0: spinner (no Flexible/Image at all) →
  complete blank. ✓

Commit: `6bf4f4c`
