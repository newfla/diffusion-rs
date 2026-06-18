# Phase 1: Flutter UI Foundation (Mock Mode) - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-18
**Phase:** 1-flutter-ui-foundation-mock-mode
**Areas discussed:** Left panel form grouping, Flutter project architecture, Preset catalog completeness, Right panel initial state

---

## Left Panel Form Grouping

| Option | Description | Selected |
|--------|-------------|----------|
| Collapsible sections | Groups: Model, Generation, Post-processing, Advanced. Each expandable/collapsible. | ✓ |
| Flat scrollable list | All 15 fields in a single column, no grouping. Simple but harder to scan. | |
| Fixed sections without collapse | Same groups but always visible. Clearer but takes more vertical space. | |

**User's choice:** Collapsible sections — with modifications: cache moves to Advanced (not Post-processing), and **no batch field**.

**Notes:** User explicitly removed batch field from Phase 1 scope.

---

| Option | Description | Selected |
|--------|-------------|----------|
| All sections expanded | User sees everything immediately. | |
| Model + Generation expanded, Post-processing + Advanced collapsed | Shows essential fields by default; advanced visible on demand. | ✓ |
| All collapsed except Model | Maximum compactness on launch. | |

**User's choice:** Model + Generation expanded, Post-processing + Advanced collapsed by default.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Seed after width/height in Generation | Order: prompt → negative → steps → width/height → seed. | ✓ |
| Seed at end of Generation | Seed as last field, visually separated. | |

**User's choice:** Seed after width/height (recommended order).

---

| Option | Description | Selected |
|--------|-------------|----------|
| Yellow inline banner under upscaler dropdown | Warning text + auto-select UCACHE. | |
| Inline warning without auto-selection | Shows message; user chooses cache manually. | |
| Warning under cache dropdown in Advanced | Warning appears in Advanced section near cache field. | ✓ |

**User's choice:** FORM-15 warning appears in Advanced section under the cache dropdown. No auto-selection.

---

## Flutter Project Architecture

| Option | Description | Selected |
|--------|-------------|----------|
| Feature-based | lib/features/params/, lib/features/generation/, lib/features/output/, lib/shared/. | ✓ |
| Layer-based | lib/presentation/, lib/domain/, lib/infrastructure/. | |
| Flat with large single files | lib/ without subdirectories. Fast for Phase 1, hard to maintain in Phase 2. | |

**User's choice:** Feature-based folder structure (recommended).

---

| Option | Description | Selected |
|--------|-------------|----------|
| Interface + provider injection | GenerationService abstract class; Mock/Rust implementations; provider decides which. Phase 2 = one line change. | ✓ |
| Compile-time flag | Dart const bool kMockMode. Less flexible. | |
| You decide | Claude picks idiomatic Riverpod approach. | |

**User's choice:** Interface + provider injection.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Shared Riverpod providers | Both panels read global providers. No prop drilling through MainScreen. | ✓ |
| Callbacks / setState in MainScreen | MainScreen holds state, passes callbacks to children. | |
| You decide | Claude picks most idiomatic approach. | |

**User's choice:** Shared Riverpod providers.

---

## Preset Catalog Completeness

| Option | Description | Selected |
|--------|-------------|----------|
| All Rust presets | Dart list mirrors src/preset.rs exactly — all 30+ presets with Weight variants. | ✓ |
| Representative subset (~8-10 presets) | SD 1.5, SDXL, Flux Schnell, Flux Dev, Chroma + 3-4 others. | |
| Flagship only (one per family) | One per family: SD1.x, SD2.x, SDXL, SD3, Flux, Chroma. | |

**User's choice:** Full catalog — all presets matching src/preset.rs.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Human-readable enum labels (Q4_K, Q8_0, F16...) | Show variant string directly. | ✓ |
| Only common weights (Q4_K, Q8_0, F16) | Filter exotic weight types. | |
| You decide | Claude chooses subset based on typical use. | |

**User's choice:** Human-readable enum labels — show all Weight variants as-is.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Hidden completely | If preset has no Weight, dropdown disappears. | |
| Visible but disabled | Dropdown stays, labeled "N/A" or greyed out. | ✓ |

**User's choice:** Weights dropdown visible but disabled when preset has no Weight variants.

---

## Right Panel Initial State

| Option | Description | Selected |
|--------|-------------|----------|
| Neutral placeholder with icon + text | Large Yaru image icon + "Configure parameters and press Generate". | ✓ |
| Empty grey area | Background only, no text. | |
| Static demo/sample image | Example image in assets to show the GUI's potential. | |

**User's choice:** Neutral placeholder with icon + instructional text.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Yaru spinner centered | YaruCircularProgressIndicator while waiting for first preview frame. | ✓ |
| Same initial placeholder | Placeholder stays until first preview frame arrives. | |
| You decide | Claude picks Yaru-idiomatic approach. | |

**User's choice:** Yaru spinner when generation starts (before first preview frame).

---

| Option | Description | Selected |
|--------|-------------|----------|
| Image stays + "Saved to..." toast (Recommended) | Image remains; Yaru SnackBar briefly shows file path. | ✓ |
| Reset to initial placeholder | Panel clears after save, ready for next generation. | |
| You decide | Claude picks appropriate UX. | |

**User's choice:** Image stays visible + Yaru SnackBar with file path.

---

## Claude's Discretion

- Exact Yaru widget choices for expansion panels, snackbar duration, spinner size
- Weight dropdown disabled-state styling
- Exact `flutter_split_view` package or custom divider implementation for panel resize
- Keyboard shortcut implementation (Shortcuts/Actions vs KeyboardListener)

## Deferred Ideas

- **Batch field** (FORM-07/GEN requirements): explicitly removed from Phase 1 by user during discussion. Was in REQUIREMENTS.md but user decided to drop it.
- **History/recall of last N prompts** (v2 UX-01): out of scope for v1.
- **Gallery output** (v2 UX-02): out of scope for v1.
- **Model download progress** (v2 MDL-01): out of scope for v1.
