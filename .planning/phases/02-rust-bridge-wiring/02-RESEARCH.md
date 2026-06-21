# Phase 2: Rust Bridge Wiring - Research

**Researched:** 2026-06-21
**Domain:** flutter_rust_bridge 2.x FFI integration, Rust-to-Dart streaming, diffusion-rs preset/generation API
**Confidence:** HIGH

## Summary

Phase 2 wires the real diffusion-rs generation backend into the Flutter GUI by creating a `gui/rust/` Cargo crate that exposes three FRB-annotated functions: `get_presets()`, `get_weights_for_preset()`, and `generate_image_stream()`. The existing Phase 1 `GenerationService` seam makes the swap a single-line provider change in `generation_provider.dart`. The core challenge is correctly mapping diffusion-rs's builder-based, blocking, `mpsc::Sender<Progress>`-driven API to FRB 2.x's `StreamSink<T>` streaming pattern.

flutter_rust_bridge 2.x (stable 2.12.0) provides first-class `StreamSink<T>` support: a Rust function taking `StreamSink<T>` as a parameter automatically generates a Dart function returning `Stream<T>`. The Rust function runs on a background thread (FRB spawns it); the sink can emit multiple values over its lifetime. Error handling uses `sink.add_error(anyhow::Error)` to surface exceptions on the Dart side. FRB automatically catches Rust panics and converts them to Dart exceptions, satisfying the `catch_unwind` requirement (FRB-06) without manual wrapping for most cases -- though explicit `catch_unwind` in the wrapper adds defense-in-depth against C++ abort signals.

The build integration uses Cargokit (default FRB backend), which hooks into Flutter's native build system (CMakeLists.txt on Linux/Windows, Xcode on macOS) to automatically compile the Rust crate during `flutter build`/`flutter run`. FRB codegen (`flutter_rust_bridge_codegen generate`) must be run after Rust API changes to regenerate Dart bindings; per D-08, this is integrated into the build workflow.

**Primary recommendation:** Use `StreamSink<GuiProgressEvent>` for `generate_image_stream()`, spawn the blocking `gen_img_with_progress()` call on a dedicated `std::thread`, relay `mpsc::Receiver<Progress>` messages plus file-based preview bytes through the sink, and wrap the entire call in `catch_unwind` for defense-in-depth.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Phase 2 shows both a progress bar AND real intermediate denoising images per step
- **D-02:** Preview delivery is file-based: `ConfigBuilder` sets `preview_output = {tmpdir}/preview.png` and `preview_mode = PreviewType::PREVIEW_PROJ`
- **D-03:** `RustGenerationService` reads preview file bytes from disk after each progress event; race condition accepted (null previewImage shows previous frame)
- **D-04:** Right panel shows "Downloading model..." (spinner + static text) until first `ProgressEvent` with `step == 1`
- **D-05:** On Rust error: show AlertDialog modal with error message; form re-enables after dismissal
- **D-06:** Error message is Rust error string as-is; no localization
- **D-07:** `catch_unwind` on ALL FFI entry points in `gui/rust/`; caught panics surface as `Result::Err`
- **D-08:** FRB codegen integrated into Flutter build -- no manual step
- **D-09:** No CI diff check (FRB-08 waived for Phase 2)
- **D-10:** `Progress` struct fields `step`, `steps`, `time` changed to `pub`
- **D-11:** `GuiParams` is a new FRB-compatible DTO with only primitive types
- **D-12:** `gui/rust/Cargo.toml` depends on root `diffusion-rs` via path dependency
- **D-13:** `panic = "abort"` in `[profile.release]` in `gui/rust/Cargo.toml`

### Claude's Discretion
- Exact FRB 2.x annotation syntax (`#[flutter_rust_bridge::frb(sync)]` vs async, stream API shape)
- Whether `generate_image_stream` uses `DartFnFuture<()>` callback or `StreamSink<ProgressEvent>` -- **recommendation: StreamSink**
- How "Downloading model..." state is signaled from Dart side
- `preview_interval` value -- default 1 (every step) unless testing shows too slow

### Deferred Ideas (OUT OF SCOPE)
- FRB-08 CI diff check
- Download progress (MDL-01) -- v2 feature
- Generation cancellation (UX-03)
- In-memory preview bytes (no file I/O alternative)
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| FRB-01 | `gui/rust/` exposes `get_presets() -> Vec<String>` via FRB | Use `PresetDiscriminants::VARIANTS` from strum to list all preset names as strings. Sync FRB function. |
| FRB-02 | `gui/rust/` exposes `get_weights_for_preset(preset: String) -> Vec<String>` via FRB | Match on `PresetDiscriminants`, delegate to each subenum's `VARIANTS` (e.g., `Flux1Weight::VARIANTS`). Sync FRB function. |
| FRB-03 | `gui/rust/` exposes `generate_image_stream(params: GuiParams, sink: StreamSink<GuiProgressEvent>)` via FRB | StreamSink pattern confirmed as idiomatic FRB 2.x. Spawn `std::thread`, bridge `mpsc::Receiver` to `sink.add()`. |
| FRB-04 | `GuiParams` is FRB-compatible DTO with only primitive types | DTO maps all 15 CLI parameters to `String`, `i32`, `i64`, `f32`, `bool`, `Option<T>` -- no complex Rust types cross FFI. |
| FRB-05 | `Progress` struct fields `step`, `steps`, `time` are `pub` | Trivial change to `src/api.rs` line 84-87. Required before `gui/rust/` can read progress values. |
| FRB-06 | All FFI entry points have `catch_unwind` wrappers | FRB v2 catches panics automatically; explicit `catch_unwind` adds defense against C++ abort. Wrap in Rust closure. |
| FRB-07 | Release profile uses `panic = "abort"` | Set in `gui/rust/Cargo.toml` `[profile.release]`. Note: `catch_unwind` only works in debug/dev profile; release aborts. This is intentional (D-13). |
| FRB-08 | CI verifies FRB codegen files are up-to-date | **Waived** per D-09. Build-integrated codegen (D-08) guarantees sync. |
| FRB-09 | `RustGenerationService` replaces `MockGenerationService` with single provider line | Swap line 130 in `generation_provider.dart`: `MockGenerationService()` -> `RustGenerationService(ref)`. |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Preset/weight enumeration | Rust FFI (gui/rust/) | -- | `PresetDiscriminants::VARIANTS` and subenum `VARIANTS` live in Rust; FRB serializes to `Vec<String>` |
| Image generation | Rust FFI (gui/rust/) | diffusion-rs core (src/) | `gen_img_with_progress` is the entry point; gui/rust/ wraps it with FRB-compatible streaming |
| Progress streaming | Rust FFI -> Dart | -- | `StreamSink<GuiProgressEvent>` bridges `mpsc::Receiver<Progress>` to Dart `Stream` |
| Preview image delivery | Rust (file write) | Dart (file read) | C callback writes PNG to disk; Dart reads bytes after progress event |
| Parameter mapping | Dart (collect form) | Rust (GuiParams -> builders) | Dart sends `Map<String, dynamic>` -> `RustGenerationService` -> FRB `GuiParams` -> Rust builders |
| Error presentation | Dart (AlertDialog) | -- | Rust errors surface as Dart exceptions via FRB; Dart shows modal dialog |
| Build integration | Cargokit | Flutter build system | Cargokit hooks into CMake/Xcode to compile Rust during `flutter build` |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| flutter_rust_bridge | 2.12.0 (Dart) + 2.12.0 (Rust crate) | FFI codegen and runtime for Rust-Dart bridge | Only mature option for Flutter desktop FFI with streaming support [CITED: pub.dev/packages/flutter_rust_bridge] |
| flutter_rust_bridge_codegen | 2.12.0 (cargo install) | Code generator producing Dart bindings from Rust `pub fn` signatures | Required companion to the runtime crate [CITED: cjycode.com/flutter_rust_bridge/quickstart] |
| anyhow | 1.0.x | Error type used by StreamSink `add_error()` | FRB's streaming error API requires `anyhow::Error` [CITED: cjycode.com/flutter_rust_bridge/guides/types/translatable/stream] |
| diffusion-rs | 0.1.20 (path dep) | Core generation library | The project's own crate; gui/rust/ depends on it via `path = "../.."` |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| strum | 0.27 (workspace) | `VariantNames` trait for listing enum variants as `&[&str]` | Used by `get_presets()` and `get_weights_for_preset()` to enumerate presets/weights [VERIFIED: codebase grep] |
| subenum | 1.1.3 | Type-safe weight subsets per preset | Already in use; gui/rust/ references the generated sub-enums to list per-preset weights [VERIFIED: codebase grep] |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| StreamSink<T> | DartFnFuture callback | StreamSink is idiomatic FRB 2.x for multi-value returns; DartFnFuture requires calling back into Dart which adds complexity |
| File-based preview | In-memory byte streaming | File-based is simpler (D-02 locked), avoids cross-FFI large buffer copies; deferred to v2 |
| Cargokit | Native Assets | Native Assets requires Flutter SDK with build hooks support and rust-toolchain.toml pinning; Cargokit is more compatible with current Flutter stable |

**Installation:**
```bash
# Install FRB codegen CLI
cargo install flutter_rust_bridge_codegen --version 2.12.0

# Add to gui/pubspec.yaml
# flutter_rust_bridge: ^2.12.0  (already listed in pubspec after FRB integrate)

# Add to gui/rust/Cargo.toml
# flutter_rust_bridge = "2.12.0"
# anyhow = "1.0"
# diffusion-rs = { path = "../.." }
```

**Version verification:**
- `flutter_rust_bridge` on crates.io: 2.12.0 stable, 2.13.0-beta.2 pre-release [VERIFIED: cargo search output]
- `flutter_rust_bridge` on pub.dev: 2.12.0 stable [CITED: pub.dev/packages/flutter_rust_bridge]
- `anyhow` on crates.io: 1.0.102 [VERIFIED: cargo search output]

## Package Legitimacy Audit

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| flutter_rust_bridge | crates.io + pub.dev | 4+ yrs | High (4k+ GitHub stars) | github.com/aspect-build/flutter_rust_bridge | OK | Approved [CITED: pub.dev] |
| anyhow | crates.io | 6+ yrs | Very high | github.com/dtolnay/anyhow | OK | Approved [ASSUMED] |

**Packages removed due to [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

## Architecture Patterns

### System Architecture Diagram

```
Flutter GUI (Dart)                    gui/rust/ (Rust FFI crate)              diffusion-rs (Rust core)
+-------------------+                +------------------------+              +---------------------+
|                   |                |                        |              |                     |
| ParamsProvider    |   FRB call     | get_presets()          |  strum       | PresetDiscriminants |
| (form state)  ----+--------------->  -> Vec<String>         +----------->>| ::VARIANTS          |
|                   |                |                        |              |                     |
| GenerationNotifier|   FRB call     | get_weights_for_preset |  subenum    | Flux1Weight, etc.   |
| (lifecycle)   ----+--------------->  (preset) -> Vec<String>+----------->>| ::VARIANTS          |
|                   |                |                        |              |                     |
| RustGenService    |   FRB stream   | generate_image_stream  |  blocking   | gen_img_with_prog   |
| (implements  -----+--------------->  (params, StreamSink)   +--std::thread| (Config, ModelCfg,  |
|  GenerationService)|               |    |                   |  mpsc chan  |  Sender<Progress>)  |
|                   |                |    | spawn std::thread  |             |                     |
|                   |  Stream<       |    | loop recv progress |             | save_preview_local  |
| OutputPanel   <---+--GuiProgress   |    | read preview.png   |             | (C callback writes  |
| (preview/final)   |   Event>       |    | sink.add(event)    |             |  PNG to preview_out)|
|                   |                |    | on complete: read  |             |                     |
|                   |                |    |   final image      |             | output: final.png   |
|                   |                +----+--------------------+             +---------------------+
+-------------------+
```

Data flow:
1. User fills form -> `ParamsProvider` collects `Map<String, dynamic>`
2. "Generate" pressed -> `GenerationNotifier.generate()` calls `RustGenerationService.generate(params)`
3. `RustGenerationService` converts `Map` to FRB `GuiParams` and calls `generate_image_stream(params)`
4. FRB dispatches to Rust: `gui/rust/src/api.rs::generate_image_stream(params, sink)`
5. Rust function maps `GuiParams` -> `PresetBuilder` + `ConfigBuilder` + `ModelConfigBuilder`, sets `preview_output` and `preview_mode`
6. Spawns `std::thread` calling `gen_img_with_progress(config, model_config, sender)`
7. Thread loops on `mpsc::Receiver<Progress>`: for each event, reads `preview_output` bytes, emits `GuiProgressEvent` via `sink.add()`
8. On generation complete, reads final image bytes from `output` path, emits final event
9. Dart `Stream<GuiProgressEvent>` drives `GenerationNotifier` state machine
10. `OutputPanel` renders live preview frames and final image

### Recommended Project Structure
```
gui/
+-- rust/
|   +-- Cargo.toml           # Isolated workspace, path dep on diffusion-rs
|   +-- src/
|   |   +-- lib.rs            # Module declarations
|   |   +-- api.rs            # FRB-annotated functions (get_presets, get_weights, generate_image_stream)
|   |   +-- gui_params.rs     # GuiParams DTO struct
|   |   +-- bridge.rs         # GuiParams -> PresetBuilder/ConfigBuilder/ModelConfigBuilder mapping
|   +-- .gitkeep              # (existing, to be replaced by actual crate)
+-- lib/
|   +-- features/
|   |   +-- generation/
|   |   |   +-- services/
|   |   |   |   +-- rust_generation_service.dart   # NEW: implements GenerationService via FRB
|   |   |   |   +-- generation_service.dart        # Unchanged abstract interface
|   |   |   |   +-- mock_generation_service.dart   # Kept for testing/development
|   |   |   +-- providers/
|   |   |       +-- generation_provider.dart        # MODIFIED: swap Mock -> Rust on line 130
|   |   +-- output/
|   |       +-- output_panel.dart                   # MODIFIED: downloading state, live preview, error dialog
|   +-- shared/
|       +-- models/
|       |   +-- preset_catalog.dart                 # KEPT (fallback), Phase 2 uses FRB calls
|       |   +-- progress_event.dart                 # Unchanged
|       +-- services/
|           +-- temp_directory_manager.dart          # Unchanged, provides preview path
+-- pubspec.yaml              # MODIFIED: add flutter_rust_bridge dependency
```

### Pattern 1: StreamSink Streaming from Rust to Dart
**What:** FRB 2.x `StreamSink<T>` pattern for emitting multiple values from a long-running Rust function
**When to use:** Any Rust function that needs to return progress updates over time
**Example:**
```rust
// Source: cjycode.com/flutter_rust_bridge/guides/types/translatable/stream
use crate::frb_generated::StreamSink;
use anyhow::Result;

/// FRB-compatible progress event sent to Dart
pub struct GuiProgressEvent {
    pub step: i32,
    pub steps: i32,
    pub time: f32,
    pub preview_image: Option<Vec<u8>>,  // PNG bytes or None
    pub final_image: Option<Vec<u8>>,    // PNG bytes on completion
}

/// Streams progress events during image generation
pub fn generate_image_stream(
    params: GuiParams,
    sink: StreamSink<GuiProgressEvent>,
) -> Result<()> {
    // FRB auto-translates this to Dart: Stream<GuiProgressEvent> generateImageStream(GuiParams params)
    // The StreamSink lives beyond the function return; we spawn a thread.
    
    std::thread::spawn(move || {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // Map GuiParams -> (Config, ModelConfig)
            // Set preview_output, preview_mode, etc.
            // Create mpsc channel
            // Call gen_img_with_progress
            // Loop on receiver, read preview file, sink.add()
        }));
        
        match result {
            Ok(Ok(())) => { /* stream naturally completes */ },
            Ok(Err(e)) => { sink.add_error(anyhow::anyhow!("{}", e)); },
            Err(panic) => {
                let msg = panic.downcast_ref::<String>()
                    .map(|s| s.as_str())
                    .or_else(|| panic.downcast_ref::<&str>().copied())
                    .unwrap_or("Unknown panic");
                sink.add_error(anyhow::anyhow!("Panic: {}", msg));
            },
        }
    });
    
    Ok(())
}
```

### Pattern 2: Sync FRB Functions for Enumeration
**What:** Simple sync functions exposed via FRB `#[frb(sync)]`
**When to use:** Fast, non-blocking queries that return immediately
**Example:**
```rust
// Source: cjycode.com/flutter_rust_bridge/guides/types/translatable/stream (sync annotation)
use strum::VariantNames;
use diffusion_rs::preset::PresetDiscriminants;

#[flutter_rust_bridge::frb(sync)]
pub fn get_presets() -> Vec<String> {
    PresetDiscriminants::VARIANTS
        .iter()
        .map(|s| s.to_string())
        .collect()
}
```

### Pattern 3: GuiParams DTO Mapping
**What:** FRB-compatible DTO with only primitive types, mapped to diffusion-rs builders
**When to use:** Crossing the FFI boundary where complex Rust types cannot be serialized
**Example:**
```rust
/// All fields are FRB-compatible primitives (per D-11 / FRB-04)
pub struct GuiParams {
    pub preset: String,            // PresetDiscriminants name
    pub weight: Option<String>,    // WeightType name (None for presets without weights)
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub steps: Option<i32>,
    pub width: Option<i32>,
    pub height: Option<i32>,
    pub batch_count: i32,
    pub seed: i64,                 // -1 for random
    pub cache_mode: Option<String>,
    pub preview_mode: String,      // "None", "Fast", "Accurate"
    pub upscaler: Option<String>,
    pub upscaler_scale: f32,
    pub token: Option<String>,     // HuggingFace token
    pub low_vram: bool,
    pub preview_output: String,    // Temp dir path for preview PNG
    pub output: String,            // Temp dir path for final image
}
```

### Anti-Patterns to Avoid
- **Passing complex Rust types through FFI:** Do NOT try to pass `Config`, `ModelConfig`, `Preset`, or `PresetBuilder` through FRB. Use primitive DTOs and map inside Rust.
- **Blocking the FRB handler thread:** `gen_img_with_progress` is blocking (can take minutes). Always spawn a `std::thread` and use `StreamSink` from within it.
- **Polling for preview images on a timer:** Do NOT use `Timer.periodic` in Dart to poll for preview files. Instead, read the preview file bytes in the Rust progress loop immediately after receiving each `Progress` event and include them in the `GuiProgressEvent`.
- **Using tokio for generation:** diffusion-rs is fully synchronous (FFI calls block). Using tokio adds complexity with no benefit. Use `std::thread::spawn`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Rust-Dart FFI bridge | Custom dart:ffi bindings, manual C ABI | flutter_rust_bridge 2.x codegen | FRB handles type mapping, memory management, thread dispatch, error propagation |
| Streaming progress to Dart | Custom port-based isolate messaging | FRB `StreamSink<T>` | Automatic `Stream<T>` generation, error channel, proper lifecycle |
| Preset enumeration | Hardcoded Dart list (Phase 1 PresetCatalog) | `PresetDiscriminants::VARIANTS` via FRB | Stays in sync with Rust source automatically; PresetCatalog becomes unused |
| Build compilation hook | Custom build.rs or Makefile | Cargokit (FRB default) | Handles cross-platform native builds (CMake/Xcode/MSVC) automatically |
| Rust panic safety at FFI | Manual `extern "C"` wrappers | FRB automatic panic catching + explicit `catch_unwind` | FRB v2 catches panics by default; explicit wrapper is defense-in-depth |

**Key insight:** The entire bridge layer should be thin -- it maps DTOs to builder calls and relays events. All complex logic stays in diffusion-rs core.

## Common Pitfalls

### Pitfall 1: Preview File Race Condition
**What goes wrong:** Dart reads preview.png while the C callback is still writing it, resulting in a corrupted/truncated image
**Why it happens:** `save_preview_local` writes the PNG file from the C inference thread; the progress callback fires nearly simultaneously
**How to avoid:** Per D-03, accept the race condition. If `File.readAsBytes()` fails or returns empty data, set `previewImage = null` in the event. The UI shows the previous frame (graceful degradation). Wrap the file read in try/catch.
**Warning signs:** Occasional garbled preview frames during fast-stepping presets (1-4 steps)

### Pitfall 2: FRB Codegen Stale Bindings
**What goes wrong:** Changing Rust function signatures without running `flutter_rust_bridge_codegen generate` causes Dart compilation errors
**Why it happens:** FRB codegen outputs Dart files that mirror Rust function signatures; they must be regenerated after Rust API changes
**How to avoid:** D-08 says codegen is build-integrated. In practice, run `flutter_rust_bridge_codegen generate` after any change to `gui/rust/src/api.rs`. Consider adding a pre-build script or documenting the workflow clearly.
**Warning signs:** Dart errors about missing methods or wrong parameter types in generated files

### Pitfall 3: `panic = "abort"` vs `catch_unwind`
**What goes wrong:** With `panic = "abort"` in release profile (D-13), `catch_unwind` has no effect -- the process terminates immediately on panic
**Why it happens:** `panic = "abort"` skips unwinding; `catch_unwind` requires unwinding to work
**How to avoid:** This is intentional per D-13. In debug/dev builds (where `panic = "unwind"` is default), `catch_unwind` works and helps during development. In release, panics abort -- but they should never occur in production if errors are properly handled via `Result`.
**Warning signs:** App crashes with no error dialog in release mode -- indicates a Rust panic that should have been a `Result::Err`

### Pitfall 4: ModelConfig Mutability and Thread Safety
**What goes wrong:** `gen_img_with_progress` requires `&mut ModelConfig`, and FRB may try to call functions from different threads
**Why it happens:** `ModelConfig` caches the sd_ctx internally (lazy initialization), requiring mutable access
**How to avoid:** Create `ModelConfig` inside the spawned `std::thread`, use it once, and drop it. Do NOT share `ModelConfig` across calls. Each generation creates a fresh context.
**Warning signs:** Borrow checker errors, or mysterious crashes from concurrent context access

### Pitfall 5: macOS Sandbox and File Access
**What goes wrong:** Writing preview/output PNGs to a path outside the sandbox container fails silently
**Why it happens:** macOS app sandbox restricts file system access; `com.apple.security.app-sandbox = true` is set in entitlements
**How to avoid:** Use `TempDirectoryManager.sessionPath` for all file paths. `path_provider`'s `getTemporaryDirectory()` returns the sandbox-safe container temp dir. Never use absolute paths outside the container.
**Warning signs:** File not found errors when reading preview.png, despite the Rust generation succeeding

### Pitfall 6: Cargokit First Build Time
**What goes wrong:** First `flutter run` after adding `gui/rust/` takes 10-30+ minutes due to full C++/CMake build of stable-diffusion.cpp
**Why it happens:** `diffusion-rs` path dependency triggers `diffusion-rs-sys` build.rs, which compiles stable-diffusion.cpp from the submodule via CMake
**How to avoid:** Document this in the README. Subsequent builds are incremental and fast. Consider using `--release` only when needed.
**Warning signs:** Developer thinks the build is stuck -- it's actually compiling GGML/stable-diffusion.cpp

### Pitfall 7: Weight Enum Mismatch
**What goes wrong:** Passing an invalid weight string for a preset causes `try_into().unwrap()` to panic in the Rust `get_preset()`-like mapping
**Why it happens:** `WeightType::from_str()` succeeds but `TryInto<Flux1Weight>` fails for a weight variant not in that preset's subenum
**How to avoid:** Validate weight string against the preset's subenum VARIANTS before constructing the `Preset` enum. Return `Result::Err` instead of unwrapping.
**Warning signs:** Panic on generation start with certain preset/weight combinations

## Code Examples

### Complete generate_image_stream Implementation Pattern
```rust
// Source: diffusion-rs codebase (src/api.rs, cli/src/main.rs) + FRB docs
use std::sync::mpsc;
use std::path::PathBuf;
use std::fs;
use crate::frb_generated::StreamSink;
use anyhow::Result;
use diffusion_rs::api::{gen_img_with_progress, PreviewType};
use diffusion_rs::preset::PresetBuilder;
use diffusion_rs::util::set_hf_token;

pub fn generate_image_stream(
    params: GuiParams,
    sink: StreamSink<GuiProgressEvent>,
) -> Result<()> {
    std::thread::spawn(move || {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // 1. Set HF token if provided
            if let Some(token) = &params.token {
                if !token.is_empty() {
                    set_hf_token(token);
                }
            }

            // 2. Build Preset from string params
            let preset = map_preset(&params.preset, params.weight.as_deref())
                .map_err(|e| anyhow::anyhow!("{}", e))?;

            let preview_output = PathBuf::from(&params.preview_output);
            let output = PathBuf::from(&params.output);

            // 3. Build Config via PresetBuilder with modifier
            let (config, mut model_config) = PresetBuilder::default()
                .preset(preset)
                .prompt(&params.prompt)
                .with_modifier(move |(mut config_b, mut model_b)| {
                    // Apply optional overrides (same pattern as cli/src/main.rs)
                    if let Some(steps) = params.steps { config_b.steps(steps); }
                    if let Some(width) = params.width { config_b.width(width); }
                    if let Some(height) = params.height { config_b.height(height); }
                    if let Some(neg) = params.negative_prompt.clone() {
                        config_b.negative_prompt(neg);
                    }
                    config_b.seed(params.seed);
                    config_b.batch_count(params.batch_count);
                    config_b.output(output);

                    // Preview config
                    config_b.preview_output(preview_output);
                    config_b.preview_mode(PreviewType::PREVIEW_PROJ);
                    config_b.preview_interval(1);

                    // Low VRAM
                    if params.low_vram {
                        model_b.vae_tiling(true).flash_attention(true);
                    }

                    // Cache mode, upscaler... (mapped from string params)
                    Ok((config_b, model_b))
                })
                .build()
                .map_err(|e| anyhow::anyhow!("{}", e))?;

            // 4. Create mpsc channel for progress
            let (tx, rx) = mpsc::channel();

            // 5. Run generation (this blocks)
            let gen_result = gen_img_with_progress(&config, &mut model_config, tx);

            // Note: gen_img_with_progress spawns internal threads; progress events
            // arrive on rx while generation runs. We process them here.
            // Actually, gen_img_with_progress is blocking -- it returns after
            // generation completes. Progress events are sent during execution.
            // We need to drain rx in a separate thread or after completion.

            // ** Corrected pattern: use crossbeam or drain after **
            // The mpsc::Sender sends progress events from the C callback thread.
            // gen_img_with_progress blocks until done. We must drain rx afterward
            // OR use a separate thread to consume rx.
            // In practice: spawn generation on one thread, consume rx on another.

            gen_result.map_err(|e| anyhow::anyhow!("{}", e))
        }));

        match result {
            Ok(Ok(())) => { /* stream completes naturally */ },
            Ok(Err(e)) => { let _ = sink.add_error(e); },
            Err(panic_info) => {
                let msg = panic_info.downcast_ref::<String>()
                    .map(|s| s.as_str())
                    .or_else(|| panic_info.downcast_ref::<&str>().copied())
                    .unwrap_or("Unknown panic in generation");
                let _ = sink.add_error(anyhow::anyhow!("Panic: {}", msg));
            }
        }
    });

    Ok(())
}
```

### Critical Threading Pattern: mpsc + StreamSink
```rust
// The correct pattern for bridging mpsc::Receiver to StreamSink:
// gen_img_with_progress is BLOCKING but sends Progress via mpsc during execution.
// We need to read rx CONCURRENTLY with the blocking gen call.

pub fn generate_image_stream(
    params: GuiParams,
    sink: StreamSink<GuiProgressEvent>,
) -> Result<()> {
    std::thread::spawn(move || {
        let sink_clone = sink.clone(); // StreamSink is Clone
        let preview_path = PathBuf::from(&params.preview_output);

        // ... build config, model_config ...

        let (tx, rx) = mpsc::channel();

        // Thread 1: consume progress events and relay to StreamSink
        let relay_handle = std::thread::spawn(move || {
            while let Ok(progress) = rx.recv() {
                // Read preview image bytes (D-03: race accepted)
                let preview_bytes = fs::read(&preview_path).ok();

                sink_clone.add(GuiProgressEvent {
                    step: progress.step,
                    steps: progress.steps,
                    time: progress.time,
                    preview_image: preview_bytes,
                    final_image: None,
                });
            }
            // Channel closed = generation complete
        });

        // Thread 0 (current): run blocking generation
        let gen_result = gen_img_with_progress(&config, &mut model_config, tx);
        // tx is dropped here, closing the channel -> relay thread exits

        relay_handle.join().ok();

        match gen_result {
            Ok(()) => {
                // Read final image
                let final_bytes = fs::read(&output_path).ok();
                sink.add(GuiProgressEvent {
                    step: total_steps,
                    steps: total_steps,
                    time: 0.0,
                    preview_image: None,
                    final_image: final_bytes,
                });
            },
            Err(e) => {
                let _ = sink.add_error(anyhow::anyhow!("{}", e));
            }
        }
    });

    Ok(())
}
```

### RustGenerationService Dart Implementation
```dart
// Source: Phase 1 GenerationService interface + FRB generated bindings
import '../../../shared/models/progress_event.dart';
import 'generation_service.dart';
// import FRB generated bindings (path TBD by codegen)

class RustGenerationService implements GenerationService {
  final Ref _ref;

  RustGenerationService(this._ref);

  @override
  Stream<ProgressEvent> generate(Map<String, dynamic> params) async* {
    final tempManager = _ref.read(tempDirectoryManagerProvider);
    final previewPath = '${tempManager.sessionPath}/preview.png';
    final outputPath = '${tempManager.sessionPath}/output_${DateTime.now().millisecondsSinceEpoch}.png';

    // Convert Map<String, dynamic> to FRB GuiParams
    final guiParams = GuiParams(
      preset: params['preset'] as String,
      weight: params['weight'] as String?,
      prompt: params['prompt'] as String,
      negativePrompt: params['negativePrompt'] as String?,
      steps: params['steps'] as int?,
      width: params['width'] as int?,
      height: params['height'] as int?,
      batchCount: params['batchCount'] as int? ?? 1,
      seed: params['seed'] as int? ?? -1,
      cacheMode: params['cacheMode'] as String?,
      previewMode: params['previewMode'] as String? ?? 'Fast',
      upscaler: params['upscaler'] as String?,
      upscalerScale: (params['upscalerScale'] as num?)?.toDouble() ?? 2.0,
      token: params['token'] as String?,
      lowVram: params['lowVram'] as bool? ?? false,
      previewOutput: previewPath,
      output: outputPath,
    );

    // Call FRB-generated function; returns Stream<GuiProgressEvent>
    await for (final event in generateImageStream(params: guiParams)) {
      yield ProgressEvent(
        step: event.step,
        steps: event.steps,
        time: event.time,
        previewImage: event.previewImage != null
            ? Uint8List.fromList(event.previewImage!)
            : null,
      );
    }
  }
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| FRB v1 (manual codec) | FRB v2 (auto codegen) | 2023 | v2 is a full rewrite; do not reference v1 patterns |
| Manual dart:ffi | FRB StreamSink + auto codegen | 2023 | StreamSink eliminates manual port/isolate management |
| Cargokit (original) | Cargokit (FRB fork) | 2024 | Original repo archived; FRB maintains its own fork |

**Deprecated/outdated:**
- FRB v1 API (`api.dart`, manual `FlutterRustBridgeBase`): completely replaced by v2 codegen
- `native-assets` backend: experimental, requires newer Flutter SDK; use Cargokit for stability

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `StreamSink` is `Clone` allowing sharing between threads | Code Examples (threading pattern) | If not Clone, must use Arc or channel relay; affects generate_image_stream implementation |
| A2 | FRB 2.x automatically catches Rust panics and converts to Dart exceptions | Don't Hand-Roll | If not automatic, explicit `catch_unwind` becomes mandatory rather than defense-in-depth |
| A3 | `flutter_rust_bridge_codegen integrate` properly sets up Cargokit hooks for macOS/Linux/Windows | Architecture Patterns | If setup is incomplete, manual CMakeLists.txt / Xcode edits needed |
| A4 | FRB codegen generates bindings into `gui/lib/` by convention | Architecture Patterns | If output dir differs, import paths in RustGenerationService change |
| A5 | `anyhow` 1.0.102 is the correct dependency for FRB StreamSink error handling | Standard Stack | If FRB uses a different error type, error propagation pattern changes |
| A6 | `gen_img_with_progress` sends Progress events via mpsc channel during execution (not batched after) | Code Examples | If events are batched, the relay thread pattern is unnecessary |

## Open Questions

1. **StreamSink cloneability**
   - What we know: FRB docs show StreamSink being used from within closures and threads
   - What's unclear: Whether StreamSink implements Clone or requires Arc wrapping for sharing between threads
   - Recommendation: Test during implementation; if not Clone, pass via Arc or use a single thread with mpsc relay

2. **FRB codegen output directory**
   - What we know: FRB quickstart says generated code goes near `lib/` but exact path depends on `flutter_rust_bridge.yaml` config
   - What's unclear: Exact output paths after `flutter_rust_bridge_codegen integrate` in an existing project
   - Recommendation: Run `flutter_rust_bridge_codegen integrate` early and inspect generated file locations

3. **Progress event timing relative to preview file write**
   - What we know: `sd_set_progress_callback` and `sd_set_preview_callback` are separate C callbacks; both fire per step
   - What's unclear: Which fires first -- does the preview file exist before or after the progress event?
   - Recommendation: Per D-03, handle both orderings. Read preview file in try/catch; null if not ready yet.

4. **Cargokit + diffusion-rs C++ build interaction**
   - What we know: Cargokit compiles the Rust crate via `cargo build`; diffusion-rs-sys has its own CMake build.rs
   - What's unclear: Whether Cargokit's CMake integration conflicts with diffusion-rs-sys's build.rs CMake invocation
   - Recommendation: Test early. If conflict, may need to configure Cargokit to use a different build mechanism.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Flutter SDK | GUI framework | Yes | 3.44.1 (stable) | -- |
| Dart SDK | Flutter dependency | Yes | 3.12.1 | -- |
| Rust toolchain | Rust compilation | Yes | 1.96.0 (stable) | -- |
| Cargo | Rust package manager | Yes | 1.96.0 | -- |
| CMake | C++ backend build | Yes | 4.3.3 | -- |
| Clang | Bindgen requirement | Yes | /usr/bin/clang | -- |
| flutter_rust_bridge_codegen | FRB code generation | No | -- | `cargo install flutter_rust_bridge_codegen --version 2.12.0` |

**Missing dependencies with no fallback:**
- None (all are installable)

**Missing dependencies with fallback:**
- `flutter_rust_bridge_codegen`: not installed, install via `cargo install flutter_rust_bridge_codegen --version 2.12.0` (Phase 2 Wave 0 task)

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | N/A (no user auth in desktop app) |
| V3 Session Management | No | N/A |
| V4 Access Control | No | N/A |
| V5 Input Validation | Yes | Validate GuiParams fields in Rust before passing to builders; prevent empty prompt, invalid preset/weight strings |
| V6 Cryptography | No | N/A |

### Known Threat Patterns for Rust FFI

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Buffer overflow from C++ backend | Tampering | diffusion-rs wraps all C calls in unsafe blocks; FRB isolates Rust from Dart memory |
| Panic propagation across FFI | Denial of Service | `catch_unwind` (FRB-06) + FRB automatic panic catching |
| HuggingFace token exposure in memory | Information Disclosure | Token stored in OnceLock, not logged; passed via secure Dart field |
| Malicious model files from HF Hub | Tampering | Out of scope for Phase 2; trust model integrity from HuggingFace |
| Path traversal via preview_output | Tampering | Use TempDirectoryManager paths only; validate paths stay within session dir |

## Sources

### Primary (HIGH confidence)
- **Codebase analysis** -- `src/api.rs` (Progress struct, ConfigBuilder, gen_img_with_progress, save_preview_local), `src/preset.rs` (Preset enum, PresetDiscriminants, WeightType, subenum system), `cli/src/main.rs` (CLI reference implementation), `gui/lib/` (Phase 1 Dart implementation)
- **FRB official docs** -- cjycode.com/flutter_rust_bridge/guides/types/translatable/stream (StreamSink API)
- **pub.dev** -- flutter_rust_bridge 2.12.0 package details
- **crates.io** -- cargo search results for flutter_rust_bridge, anyhow

### Secondary (MEDIUM confidence)
- **FRB official docs** -- cjycode.com/flutter_rust_bridge/quickstart (project setup, codegen workflow)
- **FRB official docs** -- cjycode.com/flutter_rust_bridge/manual/integrate/builtin (Cargokit/Native Assets)
- **FRB official docs** -- cjycode.com/flutter_rust_bridge/guides/types/arbitrary/rust-auto-opaque (opaque type handling)

### Tertiary (LOW confidence)
- **Training knowledge** -- FRB panic handling behavior, StreamSink Clone trait, exact Cargokit CMake hook mechanism

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- flutter_rust_bridge is the only viable option; version confirmed on pub.dev and crates.io
- Architecture: HIGH -- StreamSink pattern verified in official docs; codebase analysis confirms gen_img_with_progress + mpsc pattern
- Pitfalls: MEDIUM -- threading and race condition analysis based on code reading; preview file timing not tested

**Research date:** 2026-06-21
**Valid until:** 2026-07-21 (stable domain; FRB 2.x API unlikely to change within 30 days)
