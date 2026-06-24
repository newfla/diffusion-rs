# Architecture Patterns

**Domain:** Flutter desktop GUI wrapping a Rust AI inference library via flutter_rust_bridge
**Researched:** 2026-06-18
**Overall confidence:** MEDIUM (library patterns from training knowledge + codebase analysis; FRB2 streaming details LOW until verified against actual FRB2 changelog)

---

## Recommended Architecture

The system has three clearly separated tiers: Flutter UI, a Dart service layer, and a Rust bridge crate. The bridge crate is the only component that changes between Phase 1 (mock) and Phase 2 (real Rust). Every other layer is identical across phases.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Flutter UI Layer                            │
│                                                                     │
│   ┌──────────────────────┐        ┌──────────────────────────────┐  │
│   │   LeftPanel          │        │   RightPanel                 │  │
│   │  (GenerationForm)    │        │  (PreviewPane)               │  │
│   │  - PresetDropdown    │        │  - Image widget              │  │
│   │  - WeightsDropdown   │        │  - ProgressBar               │  │
│   │  - PromptField       │        │  - SaveButton                │  │
│   │  - NegativeField     │        └──────────────────────────────┘  │
│   │  - StepsField        │                                          │
│   │  - DimensionFields   │   ResizableSplitView (multi_split_view)  │
│   │  - BatchField        │                                          │
│   │  - CacheDropdown     │                                          │
│   │  - PreviewDropdown   │                                          │
│   │  - UpscalerDropdown  │                                          │
│   │  - TokenField        │                                          │
│   │  - SeedField         │                                          │
│   │  - StartButton       │                                          │
│   └──────────────────────┘                                          │
│                                                                     │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │                   Riverpod Providers                         │  │
│   │  GenerationParamsNotifier   GenerationNotifier (AsyncNotif.) │  │
│   │  ProgressNotifier           ThemeNotifier                    │  │
│   └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                              │  Dart service interface
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     GenerationService (abstract)                    │
│                                                                     │
│   Stream<GenerationEvent> generate(GenerationParams params)         │
│   Future<List<String>> getWeightsForPreset(String preset)           │
│   Future<List<String>> getAvailablePresets()                        │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
              ┌────────────┴─────────────┐
              │                          │
              ▼                          ▼
┌─────────────────────┐    ┌─────────────────────────────────────────┐
│  MockGenerationSvc  │    │  RustGenerationService                  │
│  (Phase 1)          │    │  (Phase 2)                              │
│                     │    │                                         │
│  Simulates progress │    │  Calls diffusion_rs_gui bridge via FRB  │
│  with fake timer    │    │  Forwards StreamSink events to Stream   │
│  Returns PNG asset  │    │  Writes output to temp dir              │
└─────────────────────┘    └──────────────────┬──────────────────────┘
                                              │  flutter_rust_bridge FFI
                                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│             gui/rust/src/lib.rs  (diffusion-rs-gui bridge)          │
│                                                                     │
│   pub fn generate_image_stream(                                     │
│       params: GuiParams,                                            │
│       sink: StreamSink<ProgressEvent>,                              │
│   )                                                                 │
│   pub fn get_presets() -> Vec<String>                               │
│   pub fn get_weights_for_preset(preset: String) -> Vec<String>      │
│                                                                     │
│   Internally: spawns std::thread, calls gen_img_with_progress()     │
│   forwarding mpsc::Receiver<Progress> into StreamSink<ProgressEvent>│
└──────────────────────────────────────────────────────────────────────┘
                              │  Cargo dependency (path = "../..")
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│             diffusion-rs library (existing, unmodified)             │
│   gen_img_with_progress(config, model_config, mpsc::Sender<Progress>│
│   Progress { step, steps, time }   DiffusionError                  │
│   Preset enum (~35 variants)       PresetDiscriminants              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Component Responsibilities

| Component | Responsibility | Location |
|-----------|---------------|----------|
| LeftPanel / GenerationForm | Renders all 15 form fields; reads from GenerationParamsNotifier; disables on generation | `gui/lib/ui/left_panel.dart` |
| RightPanel / PreviewPane | Displays current image (preview or final); shows progress bar; Save button | `gui/lib/ui/right_panel.dart` |
| ResizableSplitView | Horizontal drag-resizable splitter between panels | `multi_split_view` package |
| GenerationParamsNotifier | Holds all form values as a `GenerationParams` value object; validates cross-field constraints (upscaler requires cache); exposes typed setters | `gui/lib/providers/generation_params.dart` |
| GenerationNotifier | `AsyncNotifier<GenerationResult>` — drives lifecycle: idle → running → done/error; kicks off stream subscription on `start()`; cancels on `cancel()` | `gui/lib/providers/generation.dart` |
| ProgressNotifier | Holds latest `ProgressEvent?`; updated by GenerationNotifier as stream emits | `gui/lib/providers/progress.dart` |
| ThemeNotifier | Holds `ThemeMode` (system/light/dark); persists to SharedPreferences | `gui/lib/providers/theme.dart` |
| GenerationService (abstract) | Interface consumed by GenerationNotifier; decouples UI from backend | `gui/lib/services/generation_service.dart` |
| MockGenerationService | Implements GenerationService with a timer-based fake progress stream; returns a PNG placeholder | `gui/lib/services/mock_generation_service.dart` |
| RustGenerationService | Implements GenerationService by calling the FRB bridge; maps ProgressEvent stream; manages temp dir | `gui/lib/services/rust_generation_service.dart` |
| diffusion-rs-gui (Rust bridge crate) | Thin FRB-compatible wrapper over the diffusion-rs library; exposes GuiParams, ProgressEvent, get_presets(), get_weights_for_preset(), generate_image_stream() | `gui/rust/src/lib.rs` |
| TempDirManager | Creates app temp dir on start; recursively deletes on app exit | `gui/lib/services/temp_dir_manager.dart` |

---

## State Management: Riverpod 2.x

**Decision: Riverpod 2.x with code generation (@riverpod)**

Rationale:
- Bloc adds Event + State classes per feature, which is disproportionate for a single-screen app with one async task. The overhead is not justified.
- Provider (ChangeNotifier) lacks automatic provider disposal and has no built-in AsyncValue — you hand-roll loading/error/data states.
- Riverpod 2.x AsyncNotifier handles the generation lifecycle idiomatically: `AsyncValue<GenerationResult>` is either `AsyncLoading`, `AsyncData`, or `AsyncError` — maps directly to UI states (progress bar showing / image showing / error banner).
- `@riverpod` code generation eliminates the manual `ref.read(provider.notifier)` boilerplate.

**Provider split for this app:**

```dart
// 1. Form state — all 15 fields as one value object
@riverpod
class GenerationParamsNotifier extends _$GenerationParamsNotifier {
  @override
  GenerationParams build() => GenerationParams.defaults();

  void setPreset(String preset) {
    state = state.copyWith(preset: preset, weights: null); // reset weights on preset change
  }
  void setPrompt(String v) => state = state.copyWith(prompt: v);
  // ... one setter per field
}

// 2. Generation lifecycle
@riverpod
class GenerationNotifier extends _$GenerationNotifier {
  StreamSubscription<GenerationEvent>? _sub;

  @override
  FutureOr<GenerationResult?> build() => null; // idle

  Future<void> start() async {
    final params = ref.read(generationParamsNotifierProvider);
    final service = ref.read(generationServiceProvider);
    state = const AsyncLoading();
    _sub = service.generate(params).listen(
      (event) => ref.read(progressNotifierProvider.notifier).update(event),
      onDone: () => state = AsyncData(GenerationResult(imagePath: ...)),
      onError: (e) => state = AsyncError(e, StackTrace.current),
    );
  }

  void cancel() { _sub?.cancel(); state = AsyncData(null); }
}

// 3. Progress (current step / total steps)
@riverpod
class ProgressNotifier extends _$ProgressNotifier {
  @override
  ProgressEvent? build() => null;
  void update(GenerationEvent e) { ... }
}

// 4. Theme
@riverpod
class ThemeNotifier extends _$ThemeNotifier {
  @override
  ThemeMode build() => ThemeMode.system;
  void set(ThemeMode mode) { state = mode; }
}

// 5. Service provider (dependency injection seam for Phase 1 → Phase 2)
@riverpod
GenerationService generationService(Ref ref) => MockGenerationService();
// In Phase 2, swap to: RustGenerationService(ref.read(tempDirProvider))
```

---

## flutter_rust_bridge Async Pattern

**How FRB2 prevents UI blocking:**

FRB2 runs all bridged Rust calls on a Dart isolate worker pool. Even a synchronous Rust function called from Dart executes off the main isolate — the UI thread is never blocked. For a generation that takes minutes, this is the critical guarantee.

**FRB2 threading model:**
- Dart main isolate → spawns worker isolate(s) via FRB runtime
- Rust function executes on worker → sends result back to main isolate when done
- The C++ backend (stable-diffusion.cpp) is single-threaded per context and uses the `n_threads` param for sampling parallelism internally — this is orthogonal to FRB's isolate model

**For the GUI:** `generate_image_stream` is a Rust function returning `StreamSink<ProgressEvent>`. FRB2 converts this to a Dart `Stream<ProgressEvent>`. The Dart `GenerationNotifier` subscribes to this stream and updates UI state on each event.

---

## Progress Callback Pattern

**Problem:** The existing `gen_img_with_progress` takes an `mpsc::Sender<Progress>` (push model, Rust-internal channel). FRB2 uses `StreamSink<T>` (pull model, Dart-managed stream). `Progress` fields (`step`, `steps`, `time`) are private.

**Solution in the bridge crate (`gui/rust/src/lib.rs`):**

```rust
// New public type for FRB2 (Progress fields are private in diffusion-rs)
pub struct ProgressEvent {
    pub step: i32,
    pub steps: i32,
    pub time_per_step: f32,
}

// FRB2 sees StreamSink<ProgressEvent> and generates a Dart Stream<ProgressEvent>
pub fn generate_image_stream(params: GuiParams, sink: StreamSink<ProgressEvent>) {
    std::thread::spawn(move || {
        let (tx, rx) = std::sync::mpsc::channel::<diffusion_rs::api::Progress>();

        // Build Config and ModelConfig from GuiParams (see Data Flow below)
        let (config, mut model_config) = params.into_configs().unwrap();

        // Run blocking generation in a nested thread
        let gen_handle = std::thread::spawn(move || {
            diffusion_rs::api::gen_img_with_progress(&config, &mut model_config, tx)
        });

        // Forward progress events to Dart
        for progress in rx {
            sink.add(ProgressEvent {
                step: progress.step(),   // NOTE: needs getters added to Progress, or bridge
                steps: progress.steps(), //       uses a fork/patch of diffusion-rs
                time_per_step: progress.time(),
            });
        }

        gen_handle.join().unwrap().unwrap();
        // StreamSink closes automatically when function returns
    });
}
```

**Important constraint:** `Progress` struct fields are currently private in `src/api.rs`. Two options:
1. **Preferred:** Add `pub` to `step`, `steps`, `time` fields in `Progress` (one-line change to the Rust library, safe and non-breaking).
2. **Alternative:** Add `pub fn step(&self) -> i32`, etc. as accessor methods.

Do not fork the library. Make the minimal upstream change.

---

## Preview Image Pattern

**How previews work in the existing Rust API:**

`sd_set_preview_callback` is called with a C callback (`save_preview_local`) that writes the intermediate image to a `PathBuf`. The path is set via `config.preview_output(path)`.

**For the GUI:**

1. Before generation starts, the `RustGenerationService` creates a temp file path in the app temp dir: `$TEMP/diffusion_gui_<session_id>/preview.png`.
2. This path is embedded in `GuiParams` and passed to the Rust bridge.
3. The bridge sets `config.preview_output(preview_path)`.
4. The `ProgressEvent` does NOT carry image bytes — it only carries `step/steps/time`.
5. The Dart `RightPanel` uses a `FileImage` widget that re-reads the preview file on each `ProgressEvent` using a `Key(progress.step)` to force Flutter to re-read from disk.
6. On generation complete, the final image path is returned (also in temp dir), and the Save button copies it to the user-chosen location.

This avoids serializing image bytes over the FFI boundary on every step (which would be expensive for large images).

---

## Mock Service Pattern

**Why this is the right approach:**

The Rust build (with CMake, C++ compilation, GPU backend) takes 5–20 minutes and requires GPU toolchains. Coupling UI development to Rust build cycles would destroy iteration speed. The service interface seam allows full UI development with zero Rust dependency.

**MockGenerationService:**

```dart
class MockGenerationService implements GenerationService {
  @override
  Stream<GenerationEvent> generate(GenerationParams params) async* {
    const totalSteps = 20;
    for (int step = 1; step <= totalSteps; step++) {
      await Future.delayed(const Duration(milliseconds: 200));
      yield ProgressEvent(step: step, steps: totalSteps, timePerStep: 0.2);
    }
    // Copy placeholder PNG to temp path
    yield CompletionEvent(imagePath: await _writePlaceholder());
  }

  @override
  Future<List<String>> getAvailablePresets() async =>
      PresetData.allPresets; // hardcoded from CLI reference

  @override
  Future<List<String>> getWeightsForPreset(String preset) async =>
      PresetData.weightsFor(preset); // hardcoded lookup table
}
```

**Phase 1 → Phase 2 switch:**

Change one line in the provider file:

```dart
// Phase 1:
GenerationService generationService(Ref ref) => MockGenerationService();

// Phase 2:
GenerationService generationService(Ref ref) =>
    RustGenerationService(tempDir: ref.read(tempDirProvider));
```

All UI code, all providers, all tests remain unchanged.

---

## Two-Panel Layout: Resizable Split

**Package: `multi_split_view` (pub.dev)**

Use `MultiSplitView` with two children (left panel, right panel) and a `MultiSplitViewController` to set initial weight (e.g. 40% / 60%). The divider is drag-resizable out of the box.

```dart
MultiSplitViewTheme(
  data: MultiSplitViewThemeData(dividerThickness: 4),
  child: MultiSplitView(
    controller: MultiSplitViewController(
      areas: [Area(weight: 0.4), Area(weight: 0.6)],
    ),
    children: [LeftPanel(), RightPanel()],
  ),
)
```

**Alternative:** `split_view` or a raw `Row` with a `GestureDetector` on a vertical line. `multi_split_view` is preferred because it handles minimum area constraints and persists the split position.

---

## Rust Bridge Crate Structure

The bridge lives in `gui/rust/` as a separate Cargo crate (not as a workspace member of the root workspace — it is the Flutter project's native Rust code managed by FRB's `flutter_rust_bridge_codegen` tool).

```
gui/
  rust/
    Cargo.toml          # crate-type = ["cdylib", "staticlib"]
    src/
      lib.rs            # FRB-annotated public API
      params.rs         # GuiParams struct, into_configs() conversion
      presets.rs        # get_presets(), get_weights_for_preset()
      frb_generated.rs  # auto-generated by flutter_rust_bridge_codegen
```

**`gui/rust/Cargo.toml`:**

```toml
[package]
name = "diffusion-rs-gui"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib", "staticlib"]

[dependencies]
diffusion-rs = { path = "../..", version = "0.1.20" }
flutter_rust_bridge = "2"
```

**Key design constraint:** This bridge crate must NOT pull in the diffusion-rs GPU features at compile time unless the build script activates them. The FRB codegen tool (`flutter_rust_bridge_codegen generate`) introspects the Rust source without building the C++ backend — this requires the GPU backends to be feature-gated behind `#[cfg(feature = "cuda")]` etc., which diffusion-rs already does correctly.

---

## Rust Bridge API Surface

What the bridge crate exposes to Dart (public FRB-annotated functions):

```rust
// Preset/weight discovery (called once at startup)
pub fn get_presets() -> Vec<String>
pub fn get_weights_for_preset(preset: String) -> Vec<String>
pub fn preset_supports_weights(preset: String) -> bool

// Generation (returns via StreamSink — Dart sees this as Stream<ProgressEvent>)
pub fn generate_image_stream(params: GuiParams, sink: StreamSink<ProgressEvent>)

// Types
pub struct GuiParams {
    pub preset: String,
    pub weights: Option<String>,
    pub prompt: String,
    pub negative: Option<String>,
    pub steps: Option<u32>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub batch: u32,
    pub output_dir: String,       // always the app temp dir
    pub preview_path: String,     // temp file for preview PNG
    pub cache: Option<String>,
    pub preview: Option<String>,
    pub upscaler: Option<String>,
    pub upscaler_scale: f32,
    pub token: Option<String>,
    pub low_vram: bool,
    pub seed: i64,
}

pub struct ProgressEvent {
    pub step: i32,
    pub steps: i32,
    pub time_per_step: f32,
    pub preview_path: Option<String>, // set when a new preview PNG was written
}
```

The `params.rs` module converts `GuiParams` into `(Config, ModelConfig)` using diffusion-rs builders, mirroring what `cli/src/main.rs` does. This conversion is the most complex part of the bridge — the CLI logic is the canonical reference.

---

## Data Flow: Generation Request

```
User clicks "Start"
        │
        ▼
GenerationNotifier.start()
  reads GenerationParamsNotifier.state → GenerationParams
  sets state = AsyncLoading()
  calls generationService.generate(params) → Stream<GenerationEvent>
        │
        ▼ (Phase 1)                         ▼ (Phase 2)
MockGenerationService                  RustGenerationService
  yields ProgressEvent every 200ms       calls bridge.generateImageStream(guiParams)
  yields CompletionEvent with asset       Stream<ProgressEvent> flows from Rust → FRB → Dart
        │                                          │
        └──────────────┬───────────────────────────┘
                       ▼
        Stream<GenerationEvent> (Dart)
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
ProgressNotifier.update(event)    on completion:
  state = event                     GenerationNotifier.state = AsyncData(result)
        │                             RightPanel shows final image + Save button
        ▼
ProgressBar widget rebuilds (ref.watch)
RightPanel reloads FileImage with Key(event.step) to force re-read of preview.png
```

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: Calling gen_img_with_progress on the Dart main thread
**What goes wrong:** The Rust function blocks for minutes. If called without FRB's isolate mechanism (e.g. via dart:ffi directly), the UI freezes.
**Instead:** Always go through the FRB bridge. FRB's worker isolate handles thread safety.

### Anti-Pattern 2: One Riverpod provider per form field
**What goes wrong:** 15 separate providers for 15 form fields. Every field change triggers a broad rebuild. Cross-field validation (upscaler requires cache) requires reading 2+ providers in a notifier, creating implicit dependencies that are hard to test.
**Instead:** One `GenerationParams` value class with all fields, one `GenerationParamsNotifier`. All validation lives in one place.

### Anti-Pattern 3: Passing image bytes over FFI for every preview step
**What goes wrong:** SDXL at 1024x1024 = 3MB RGB per preview frame. At 20 steps, that is 60MB serialized over FFI. Causes latency spikes and GC pressure in Dart.
**Instead:** Write preview PNG to a temp file path (which the Rust C callback already does), and reload from disk in Flutter using `FileImage` with a step-keyed `Key` to invalidate the cache.

### Anti-Pattern 4: Making the bridge crate a workspace member of the root Cargo workspace
**What goes wrong:** FRB's codegen tool expects a standalone crate it can build without activating GPU features. If the GUI bridge is pulled into the root workspace, cargo will try to compile stable-diffusion.cpp (which takes 20+ minutes and requires CUDA/Metal) every time FRB regenerates bindings.
**Instead:** `gui/rust/` is its own isolated Cargo workspace (`[workspace]` in its own `Cargo.toml`), excluded from the root workspace. The dependency on diffusion-rs uses a path reference.

### Anti-Pattern 5: Hardcoding preset/weight lists in Dart
**What goes wrong:** The Rust library adds a new preset, and the Dart mock list is stale. The dropdown shows old presets in Phase 1 and correct presets in Phase 2 — inconsistency undermines testing.
**Instead:** Even in Phase 1, derive the mock's preset list from a single Dart constant file (`preset_data.dart`) that is manually synced from the Rust `Preset` enum. In Phase 2, query the bridge's `get_presets()` at startup. Keep the two in sync via a comment linking to the Rust source.

### Anti-Pattern 6: Forgetting to clean the temp dir on app exit
**What goes wrong:** Each generation writes 1–10 MB of PNG files to temp. After 100 runs, several hundred MB accumulate.
**Instead:** `TempDirManager` registers a `WidgetsBindingObserver` and deletes the session temp dir in `didRequestAppExit()`. Also run cleanup at startup to catch any leftover dirs from crashes.

---

## Scalability Considerations

| Concern | Current scope (v1) | Notes |
|---------|-------------------|-------|
| Concurrent generations | One at a time (by design) | ModelConfig holds mutable sd_ctx; concurrent calls would corrupt state. FRB bridge must enforce single-call semantics via a Mutex or by checking isRunning before starting. |
| Multiple output images (batch > 1) | Supported in Rust | GUI should display all batch images in RightPanel (scrollable list). |
| Large model loading (first run) | Model is loaded lazily on first gen_img call | The bridge can emit a "loading model" progress phase before the diffusion steps begin. |
| App restart with preserved settings | Not in scope for v1 | SharedPreferences can persist GenerationParams easily when needed. |

---

## Suggested Build Order

Build in this sequence to validate assumptions earliest and defer the hardest integration work:

1. **Flutter project scaffold** — `flutter create gui`, add dependencies (Riverpod, multi_split_view, Yaru), configure desktop targets.
2. **GenerationParams value class + MockGenerationService** — pure Dart, no Rust, no UI. Testable in isolation.
3. **Provider layer** — GenerationParamsNotifier, GenerationNotifier, ProgressNotifier, ThemeNotifier. Wire to mock service.
4. **Two-panel layout** — ResizableSplitView, LeftPanel shell, RightPanel shell. No logic yet.
5. **LeftPanel: all form fields** — driven by GenerationParamsNotifier. Full cross-field validation (upscaler/cache dependency, weights visibility).
6. **RightPanel: progress bar + image display** — driven by ProgressNotifier + GenerationNotifier.
7. **TempDirManager** — app lifecycle cleanup.
8. **End-to-end Phase 1 smoke test** — Click Start, watch mock progress bar, see placeholder image, click Save.
9. **FRB bridge crate scaffold** — `flutter_rust_bridge_codegen create` in `gui/rust/`, wire get_presets(), get_weights_for_preset() (no GPU, fast to compile).
10. **GuiParams → Config/ModelConfig conversion** — params.rs mirrors cli/src/main.rs logic. Unit-testable in pure Rust.
11. **generate_image_stream with StreamSink** — the FFI streaming integration. Requires GPU build. This is the Phase 2 milestone.
12. **RustGenerationService swap-in** — change one provider line. All UI tests continue to pass against mock.

---

## Component Boundaries (Summary)

```
Phase 1 build boundary (no GPU required):
┌──────────────────────────────────────────────────────────────┐
│  Flutter UI  +  Riverpod providers  +  MockGenerationService │
└──────────────────────────────────────────────────────────────┘

Phase 2 addition (requires GPU toolchain):
┌──────────────────────────────────────────────────────────────┐
│  RustGenerationService  →  FRB bridge  →  diffusion-rs lib   │
└──────────────────────────────────────────────────────────────┘
```

The service interface is the seam. Everything above it is Phase 1. Everything below it is Phase 2.

---

## Sources

- diffusion-rs codebase analysis: `src/api.rs` (gen_img_with_progress, Progress struct, preview callback), `src/preset.rs` (Preset enum variants), `cli/src/main.rs` (parameter wiring reference) — HIGH confidence
- flutter_rust_bridge 2.x patterns: training knowledge (FRB2 isolate model, StreamSink<T> API, codegen workflow) — MEDIUM confidence; verify StreamSink<T> exact signature against FRB2 changelog before implementation
- Riverpod 2.x AsyncNotifier pattern: training knowledge — MEDIUM confidence
- multi_split_view package: training knowledge — MEDIUM confidence; verify current API against pub.dev before implementation
- Progress struct fields private: confirmed from codebase (`src/api.rs` lines 85-87 show no `pub`) — HIGH confidence
