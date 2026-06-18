# Phase 1: Flutter UI Foundation (Mock Mode) - Context

**Gathered:** 2026-06-18
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver a complete two-panel Flutter desktop GUI in mock mode — all 15 CLI fields (minus batch), collapsible form sections, mock generation service with Stream-based progress, preview placeholder, image saving — with zero Rust/GPU dependencies. No flutter_rust_bridge integration yet; the seam is a Dart `GenerationService` interface that MockGenerationService implements.

</domain>

<decisions>
## Implementation Decisions

### Left Panel Form Layout
- **D-01:** Collapsible sections using Yaru expansion panels. No batch field (removed from scope).
- **D-02:** Four sections: **Model** (preset dropdown + weights dropdown), **Generation** (prompt multiline, negative, steps, width, height, seed+dice-button), **Post-processing** (preview dropdown, upscaler dropdown, upscaler_scale field), **Advanced** (cache dropdown, token password field, low_vram toggle).
- **D-03:** Default state on app launch: **Model + Generation expanded**, Post-processing + Advanced collapsed.
- **D-04:** Field order within Generation: prompt → negative → steps → width/height → seed (dice button resets to -1).
- **D-05:** FORM-15 warning (upscaler active but cache = None): shown as inline text **under the cache dropdown inside Advanced** section. No auto-selection of cache; user must choose manually.
- **D-06:** Presets without Weight variants: weights dropdown is **visible but disabled** (label "N/A" or greyed-out).

### Flutter Project Architecture
- **D-07:** Feature-based folder structure under `gui/lib/`:
  ```
  lib/
    main.dart
    app.dart
    features/
      params/
        params_panel.dart
        sections/
          model_section.dart
          generation_section.dart
          postproc_section.dart
          advanced_section.dart
        providers/
          params_provider.dart
      generation/
        providers/
          generation_provider.dart
        services/
          generation_service.dart      ← abstract interface
          mock_generation_service.dart ← Phase 1 implementation
      output/
        output_panel.dart
        providers/
          output_provider.dart
    shared/
      theme/
        theme_provider.dart
      widgets/
  ```
- **D-08:** The Phase 1 → Phase 2 seam: `GenerationService` is an abstract Dart class/interface. `MockGenerationService` implements it for Phase 1. `RustGenerationService` (Phase 2) will replace it via a **single provider line change** — no structural refactor needed.
- **D-09:** Left panel ↔ right panel communication via **shared Riverpod providers** only. No prop drilling through `MainScreen`. Consistent with Riverpod 2.x AsyncNotifier pattern.

### Preset Catalog (Dart Mock)
- **D-10:** The Dart hardcoded list replicates **all presets from `src/preset.rs`** — full catalog, same variants and Weight sub-enums as the Rust source at the time of Phase 1 build.
- **D-11:** Weight dropdown labels use the **human-readable enum string** directly (e.g., Q4_K, Q8_0, F16, F32). No filtering of exotic weight types (IQ1_M, I64, etc.) — show all available for each preset.

### Right Panel States
- **D-12:** **Initial state** (before first generation): neutral placeholder — large Yaru/Material image icon + text "Configure parameters and press Generate". Color matches active theme.
- **D-13:** **During generation, before first preview frame**: centered Yaru `YaruCircularProgressIndicator` (spinner), not the placeholder.
- **D-14:** **After image save**: image remains visible in the panel + Yaru `SnackBar` briefly showing "Saved to /path/to/file.png". Panel does not reset.

### Claude's Discretion
- Exact Yaru widget choices for expansion panels, snackbar duration, and spinner size — Claude picks what is most idiomatic for Yaru 6.x.
- Weight dropdown "disabled" state styling when preset has no Weight variants.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Rust Source (preset catalog reference)
- `src/preset.rs` — Full Preset enum and all Weight sub-enums. The Dart hardcoded catalog MUST mirror this exactly. Read lines 245+ for enum variants.
- `src/api.rs` lines 84-88 — `Progress` struct (fields `step`, `steps`, `time` are currently private; Phase 2 will make them `pub` per FRB-05, but Phase 1 does not touch this).

### Requirements & Roadmap
- `.planning/REQUIREMENTS.md` — 46 v1 requirements. Phase 1 covers SETUP-01 through MOCK-04. Authoritative source for all acceptance criteria.
- `.planning/ROADMAP.md` §Phase 1 — Success criteria (5 items) define done for this phase.
- `.planning/PROJECT.md` — Core value, constraints, 15-parameter table, tech stack decisions.

### No external specs — requirements fully captured in decisions above and in REQUIREMENTS.md.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/preset.rs` — Preset enum is the source of truth for the Dart mock catalog. Phase 1 reads it at build time to inform the hardcoded list; no codegen needed yet.
- `src/api.rs` Progress struct — defines the `step`, `steps`, `time` shape that `MockGenerationService` stream events should mirror.

### Established Patterns
- **No existing Flutter/Dart code** — `gui/` directory does not exist yet. This is a greenfield Flutter project inside the monorepo.
- Rust workspace root `Cargo.toml` must NOT include `gui/rust/` as a member (constraint SETUP-02 / STATE.md key decisions) — avoids triggering CMake/GPU build on every `cargo build`.

### Integration Points
- `gui/` lives as a subdirectory of the monorepo root. The Flutter project is self-contained. No Rust build toolchain required for Phase 1.
- Phase 2 integration point: `gui/rust/Cargo.toml` (separate workspace) will expose `get_presets()`, `get_weights_for_preset()`, `generate_image_stream()` via flutter_rust_bridge. The Dart seam is `GenerationService` abstract class (D-08).

</code_context>

<specifics>
## Specific Ideas

- **Mock stream behavior**: `MockGenerationService` emits progress events via a Dart `Stream<ProgressEvent>`, not `Timer.periodic`. The stream completes in ~5 seconds with realistic step increments.
- **Seed field**: numeric input + dice icon button that sets value to -1 (random). The -1 value means "random" to the backend (carried from CLI semantics).
- **Token field**: `YaruPasswordField` (built-in Yaru widget with visibility toggle) — no custom implementation needed.
- **Resizable panels**: `gui/` uses a drag handle (e.g., `flutter_split_view` or a custom `GestureDetector` on a divider) for horizontal panel resize. Exact package TBD by planner.
- **Keyboard shortcut**: Cmd/Ctrl+Enter triggers "Generate" — implement via `Focus` + `KeyboardListener` or `Shortcuts`/`Actions` framework in Flutter.

</specifics>

<deferred>
## Deferred Ideas

- **Batch field** (FORM-07): explicitly removed from Phase 1 scope per user decision during discussion. May be re-evaluated in a future milestone if needed.
- **History/recall of last N prompts** (v2 UX-01): out of scope for v1.
- **Gallery output panel** (v2 UX-02): out of scope for v1.
- **Download progress for models** (v2 MDL-01): out of scope for v1.

</deferred>

---

*Phase: 1-flutter-ui-foundation-mock-mode*
*Context gathered: 2026-06-18*
