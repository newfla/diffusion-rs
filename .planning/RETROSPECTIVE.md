# Retrospective: diffusion-rs GUI

---

## Milestone: v1.0 MVP

**Shipped:** 2026-06-23
**Phases:** 2 | **Plans:** 5 | **Commits:** 44
**Timeline:** 6 days (2026-06-18 → 2026-06-23)

### What Was Built

1. Flutter desktop app in gui/ with two-panel Yaru layout and mock generation service (Phase 1 Plan 01, 24 min)
2. Complete 14-field parameter form with 41-preset catalog mirroring src/preset.rs (Phase 1 Plan 02, 24 min)
3. Session-isolated temp directory lifecycle with OS-native save flow (Phase 1 Plan 03, 7 min)
4. gui/rust/ Cargo crate with FRB functions, GuiParams DTO, catch_unwind, Progress pub fields (Phase 2 Plan 01, 8 min)
5. Cargokit build integration, RustGenerationService with live preview streaming, provider swap (Phase 2 Plan 02, 13 min)

### What Worked

- **Mock-first sequencing**: Building Phase 1 with a full mock before touching Rust was the right call. It let the form, layout, and save flow be tested end-to-end with zero build chain friction. Phase 2 was then purely about wiring, not UI iteration.
- **GenerationService abstract seam**: The abstract `GenerationService` with `Stream<ProgressEvent> generate()` made Phase 2 trivially correct — swapping Mock to Rust was literally one line in the provider (FRB-09).
- **Exhaustive match arms**: Using exhaustive (no catch-all) match on PresetDiscriminants in the Rust bridge gives compile-time safety when new presets are added to diffusion-rs. Zero runtime surprises.
- **GSD workflow discipline**: The pre-planning research phase for Phase 2 surfaced the cargokit + FRB integration pattern clearly before any code was written.

### What Was Inefficient

- **FRB codegen in worktree**: The flutter_rust_bridge_codegen generate step requires `cargo expand` which triggers the full C++ build of stable-diffusion.cpp. This cannot run in a worktree/CI context without a full environment. Manual binding stubs were created and then regenerated after the build succeeded locally. Detected during Phase 2 UAT when the app showed a black screen (stem mismatch).
- **Cargokit package name discovery**: The root cause of the build failure (Cargo package name `diffusion-rs-gui` vs pod target `rust_lib_diffusion_rs_gui`) required reading deep into cargokit's `build_pod.dart` and `artifacts_provider.dart` source to understand. This is a gotcha that should be documented upfront for any flutter_rust_bridge project on macOS.
- **Missing linker flags**: `-lc++ -framework Accelerate` were not included in the initial podspec. Discovered only after the build succeeded but linking failed with undefined C++ stdlib and BLAS symbols. Should be part of any cargokit podspec that wraps a Rust crate depending on stable-diffusion.cpp.

### Patterns Established

- **Cargokit naming rule**: Cargo package name must equal the CocoaPods pod target name exactly (underscores, no hyphens). The pod target is `rust_lib_XXX`; the Cargo `[package] name` must be `rust_lib_XXX`.
- **FRB stem regeneration**: After any Cargo package rename, always re-run `flutter_rust_bridge_codegen generate` to update the `stem` in `frb_generated.dart`. Stem mismatch causes a black screen (library not found) with no obvious error message.
- **podspec linker flags for stable-diffusion.cpp**: `OTHER_LDFLAGS[sdk=macosx*]` must include `-lc++ -framework Accelerate`. Without them, hundreds of undefined symbol errors from C++ stdlib and Apple Accelerate.
- **Two-thread StreamSink relay**: The generate_image_stream pattern — worker thread calls `gen_img_with_progress`, relay thread bridges `mpsc::Receiver<Progress>` to `StreamSink<GuiProgressEvent>` — is the correct FRB 2.x streaming idiom. `executeNormal` (not `executeStream`) handles the sink port serialization.
- **previewBytes in-memory display**: Pass `previewBytes: Uint8List?` in state instead of writing preview to file and reading back. Cleaner and faster for per-step preview updates.
- **DropdownButton + InputDecorator pattern**: Flutter 3.44.x deprecated `DropdownButtonFormField.value`; use `DropdownButton` inside `InputDecorator` + `DropdownButtonHideUnderline` for equivalent styling with controlled Riverpod state.

### Key Lessons

1. **Name your Cargo crate after the pod target from the start.** The pod target name comes from `flutter_rust_bridge_codegen integrate` and is `rust_lib_{yourname}`. Set `[package] name = "rust_lib_{yourname}"` in Cargo.toml immediately. Renaming later requires regenerating FRB bindings.
2. **Document linker flags in the podspec template.** Every project using stable-diffusion.cpp on macOS will need `-lc++ -framework Accelerate`. Add to `rust_builder/macos/*.podspec` as part of the cargokit scaffold.
3. **FRB codegen needs a live build environment.** The CI diff-check (FRB-08) is the right pattern, but the first generation of bindings must happen locally with a full build chain. Plan for this as a developer setup step.
4. **catch_unwind is defense-in-depth, not the only safety.** The FRB runtime itself is resilient to errors returned as Err variants. catch_unwind guards against panics; the real reliability comes from returning proper Results from generate_image_stream.

---

## Cross-Milestone Trends

| Metric | v1.0 |
|--------|------|
| Phases | 2 |
| Plans | 5 |
| Commits | 44 |
| LOC (project) | ~4,782 |
| Timeline | 6 days |
| Requirements validated | 45/46 |
| UAT pass rate | 15/16 (1 skipped) |
| Build failures before green | 3 (package name, linker, FRB stem) |
