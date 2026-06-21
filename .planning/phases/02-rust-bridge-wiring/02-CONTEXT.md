# Phase 2: Rust Bridge Wiring - Context

**Gathered:** 2026-06-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Wire the real diffusion-rs generation backend into the Flutter GUI by implementing `RustGenerationService` via flutter_rust_bridge 2.x, replacing `MockGenerationService` through a single provider line change. The phase delivers:
- `gui/rust/` Cargo crate exposing `get_presets()`, `get_weights_for_preset()`, `generate_image_stream()` via FRB
- `RustGenerationService` in Dart consuming the FRB bindings
- Live preview images per step (file-based, PREVIEW_PROJ)
- Real final image from diffusion-rs in the right panel
- Graceful error handling (Rust panics caught via `catch_unwind`, errors shown as modal dialog)

No new UI fields or layout changes. The seam architecture from Phase 1 (D-08) makes this a purely internal swap.

</domain>

<decisions>
## Implementation Decisions

### Preview Frames per Step (intermediate images)

- **D-01:** Phase 2 shows **both** a progress bar AND real intermediate denoising images in the right panel per step. Not mock placeholders — actual frames from the diffusion process.
- **D-02:** Preview delivery is **file-based**: `ConfigBuilder` sets `preview_output = {tmpdir}/preview.png` and `preview_mode = PreviewType::PREVIEW_PROJ` (Fast, per CLI `PreviewMode::Fast` → `PREVIEW_PROJ`). The existing `save_preview_local` C callback writes the PNG to disk each step.
- **D-03:** `RustGenerationService` reads the preview file bytes from disk **after each progress event** and includes them as `previewImage: Uint8List?` in `ProgressEvent`. Read-after-write race condition is accepted — if the file isn't ready yet, `previewImage` is null and the UI shows the previous frame (graceful degradation, no crash).
- **D-04:** The right panel shows "Downloading model..." (spinner + static text) until the first `ProgressEvent` with `step == 1` arrives. This covers the case where diffusion-rs downloads the model before starting inference. No download progress tracking (v2 MDL-01).

### Error Handling UX

- **D-05:** On **Rust error** (DiffusionError returned or panic caught by FRB-06 `catch_unwind`): show an **AlertDialog modale** with the error message. The form re-enables after dismissal. No SnackBar for errors — modal ensures the user sees it.
- **D-06:** The error message displayed in the dialog is the Rust error string as-is (e.g., "Forward: out of memory" from DiffusionError). No localization or prettification in Phase 2.
- **D-07:** `catch_unwind` (FRB-06) is on ALL FFI entry points in `gui/rust/`. Caught panics surface as `Result::Err` in Dart, handled identically to DiffusionError (same dialog).

### FRB Codegen Workflow

- **D-08:** FRB codegen (`flutter_rust_bridge_codegen generate`) is **integrated into the Flutter build** — no manual step required. Developers do not need to run codegen separately.
- **D-09:** **No CI diff check** (FRB-08 requirement waived for Phase 2). The build integration guarantees sync. This simplifies CI; FRB-08 can be revisited in a future milestone if drift becomes an issue.

### Rust-Side Changes Required

- **D-10:** `Progress` struct fields `step`, `steps`, `time` in `src/api.rs` are changed to `pub` (FRB-05 prerequisite).
- **D-11:** `GuiParams` is a new FRB-compatible DTO in `gui/rust/src/` with `String`, `i32`, `i64`, `f32`, `bool`, `Option<T>` fields only — mirrors all 15 CLI parameters (FRB-04).
- **D-12:** `gui/rust/Cargo.toml` depends on the root `diffusion-rs` crate via path dependency (`diffusion-rs = { path = "../.." }`). This triggers the full C++/CMake/GPU build when building `gui/rust/` — expected and required for Phase 2.
- **D-13:** `panic = "abort"` in `[profile.release]` in `gui/rust/Cargo.toml` (FRB-07).

### Claude's Discretion

- Exact FRB 2.x annotation syntax (`#[flutter_rust_bridge::frb(sync)]` vs async, stream API shape).
- Whether `generate_image_stream` uses `DartFnFuture<()>` callback or `StreamSink<ProgressEvent>` — pick the idiomatic FRB 2.x pattern for streaming.
- How "Downloading model..." state is signaled from Dart side (e.g., timeout before first event, or explicit DownloadingEvent type).
- `preview_interval` value — default 1 (every step) unless testing shows it's too slow.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Rust API — generation and preview
- `src/api.rs` lines 83-88 — `Progress` struct (fields need `pub` per D-10)
- `src/api.rs` lines 1512-1555 — `save_preview_local` C callback + `sd_set_preview_callback` + `sd_set_progress_callback` — this is exactly how preview files and progress events are wired
- `src/api.rs` lines 988-1004 — `ConfigBuilder` fields: `preview_output`, `preview_mode`, `preview_interval`, `preview_noisy`

### CLI preview reference implementation
- `cli/src/main.rs` lines 188-191 — how CLI maps `PreviewMode::Fast` → `PreviewType::PREVIEW_PROJ` and `PreviewMode::Accurate` → `PREVIEW_VAE`
- `cli/src/main.rs` lines 236 + 252-253 — how CLI sets `preview_output` path

### Phase 1 seam architecture
- `.planning/phases/01-flutter-ui-foundation-mock-mode/01-CONTEXT.md` §D-08 — `GenerationService` abstract seam design
- `gui/lib/features/generation/services/generation_service.dart` — abstract interface `RustGenerationService` must implement
- `gui/lib/shared/models/progress_event.dart` — `ProgressEvent` model with `previewImage: Uint8List?` (already ready for Phase 2)

### Requirements
- `.planning/REQUIREMENTS.md` §FRB-01 through FRB-09 — the 9 bridge requirements for Phase 2
- `.planning/ROADMAP.md` §Phase 2 — Success Criteria (4 items define done)

### FRB 2.x documentation
- flutter_rust_bridge 2.x official docs (research agent should fetch current docs) — especially: streaming API (`StreamSink`), `catch_unwind` integration, build-time codegen integration, Flutter-side bindings

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `gui/lib/features/generation/services/generation_service.dart` — abstract `GenerationService` interface; `RustGenerationService` implements exactly this.
- `gui/lib/shared/models/progress_event.dart` — `ProgressEvent` with `previewImage: Uint8List?` field already present; null-safe for Phase 1, filled in Phase 2.
- `gui/lib/features/generation/services/mock_generation_service.dart` — reference for how `GenerationService.generate()` stream should behave; `RustGenerationService` replaces it.
- `gui/lib/shared/services/temp_directory_manager.dart` — session temp directory already available; preview PNG written to `{tmpDir}/preview.png`.

### Established Patterns
- **Riverpod 2.x AsyncNotifier** — `GenerationProvider` (generation_provider.dart) consumes the service stream; no structural change needed, just wire new service.
- **Feature-based folder structure** — new `gui/rust/` crate lives outside `gui/lib/` (it's a Cargo crate, not a Dart lib). The generated Dart bindings land in `gui/lib/` per FRB 2.x convention.
- **Isolated Cargo workspace** — `gui/rust/Cargo.toml` is NOT a member of root workspace (SETUP-02 already done conceptually; `gui/rust/` directory doesn't exist yet and must be created).

### Integration Points
- `src/api.rs` — `Progress` struct must have `pub` fields before `gui/rust/` can re-export them via FRB (D-10 / FRB-05).
- `gui/lib/features/generation/providers/generation_provider.dart` — find the single line that instantiates `MockGenerationService` and replace with `RustGenerationService` (FRB-09).

</code_context>

<specifics>
## Specific Ideas

- **Preview mode:** Use `PreviewType::PREVIEW_PROJ` (equivalent to CLI `--preview fast`). User explicitly called out this combination: `preview_output` + `PreviewType::PREVIEW_PROJ`.
- **"Downloading model..." state:** Triggered in Dart when the generation stream hasn't emitted its first event after a short timeout (e.g., 2 seconds). The right panel shows the Yaru spinner with static text "Downloading model..." instead of the normal idle placeholder.
- **Error dialog title:** Keep it simple — "Generation Failed" with the Rust error string as body text and a single "OK" button.

</specifics>

<deferred>
## Deferred Ideas

- **FRB-08 CI diff check** — waived for Phase 2 per D-09. Can be revisited if build-integrated codegen proves unreliable.
- **Download progress (MDL-01)** — v2 feature. Phase 2 shows "Downloading model..." static text only.
- **Generation cancellation (UX-03)** — requires abort signal in C++ backend; v2.
- **In-memory preview bytes (no file I/O)** — alternative to file-based approach; considered and deferred. File-based is simpler and sufficient.

</deferred>

---

*Phase: 2-rust-bridge-wiring*
*Context gathered: 2026-06-21*
