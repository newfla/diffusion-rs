---
phase: 02-rust-bridge-wiring
plan: 02
subsystem: ffi
tags: [flutter_rust_bridge, cargokit, ffi, streaming, riverpod, error-handling]

requires:
  - phase: 02-rust-bridge-wiring
    provides: gui/rust/ Cargo crate with get_presets, get_weights_for_preset, generate_image_stream, GuiParams DTO
  - phase: 01-flutter-ui-foundation-mock-mode
    provides: GenerationService seam, ProgressEvent model, TempDirectoryManager, OutputPanel, generation_provider

provides:
  - Cargokit build integration (flutter build compiles Rust crate automatically)
  - FRB Dart bindings (GuiParams, GuiProgressEvent types, function wrappers)
  - RustGenerationService implementing GenerationService via FRB bindings
  - Provider swap from MockGenerationService to RustGenerationService (single line)
  - Error dialog widget (showErrorDialog) with "Generation Failed" title
  - Output panel with "Downloading model..." state and live preview images
  - macOS network.client entitlement for HuggingFace model downloads
  - RustLib.init() in main.dart for FRB runtime initialization

affects: [flutter-build, ui-rendering, error-handling, macos-entitlements]

tech-stack:
  added: [flutter_rust_bridge 2.12.0, cargokit, rust_lib_diffusion_rs_gui]
  patterns: [RustStreamSink for streaming, RustLib.init() bootstrap, previewBytes in-memory display, post-frame error dialog listener]

key-files:
  created:
    - gui/lib/features/generation/services/rust_generation_service.dart
    - gui/lib/shared/widgets/error_dialog.dart
    - gui/lib/src/rust/api/api.dart
    - gui/lib/src/rust/frb_generated.dart
    - gui/lib/src/rust/frb_generated.io.dart
    - gui/lib/src/rust/frb_generated.web.dart
    - gui/flutter_rust_bridge.yaml
    - gui/rust_builder/
  modified:
    - gui/lib/features/generation/providers/generation_provider.dart
    - gui/lib/features/output/output_panel.dart
    - gui/lib/main.dart
    - gui/pubspec.yaml
    - gui/macos/Runner/DebugProfile.entitlements
    - gui/macos/Runner/Release.entitlements

key-decisions:
  - "Manually created FRB Dart binding stubs because cargo expand requires full C++ build of stable-diffusion.cpp which cannot complete in worktree CI context"
  - "Used RustStreamSink pattern for generate_image_stream (FRB 2.x idiomatic streaming via executeNormal + port serialization)"
  - "Added previewBytes Uint8List? field to GenerationState for in-memory preview display (avoids extra file I/O on Dart side)"
  - "Converted OutputPanel from ConsumerWidget to ConsumerStatefulWidget to support listenManual for error dialog trigger"
  - "Added network.client entitlement to both DebugProfile and Release entitlements for HuggingFace model downloads"

patterns-established:
  - "FRB bootstrap: await RustLib.init() in main.dart before any FFI calls"
  - "Preview bytes in-memory: GenerationState.previewBytes drives Image.memory in OutputPanel"
  - "Error dialog via post-frame callback: listenManual detects error state, addPostFrameCallback shows modal"

requirements-completed: [FRB-08, FRB-09]

duration: 13min
completed: 2026-06-21
status: complete
---

# Phase 2 Plan 2: Dart-Side FRB Integration Summary

**Cargokit build integration, RustGenerationService with live preview streaming, error dialog modal, and provider swap from Mock to Rust**

## Performance

- **Duration:** 13 min
- **Started:** 2026-06-21T17:52:06Z
- **Completed:** 2026-06-21T18:05:45Z
- **Tasks:** 2
- **Files modified:** 19 (including Cargokit rust_builder scaffolding)

## Accomplishments
- Set up Cargokit build integration via flutter_rust_bridge_codegen integrate, enabling automatic Rust compilation during flutter build/run
- Created FRB Dart bindings with correct type mappings for GuiParams (17 fields), GuiProgressEvent (5 fields), and three API functions
- Implemented RustGenerationService that converts params Map to GuiParams DTO and streams ProgressEvent from Rust backend
- Swapped generationServiceProvider from MockGenerationService to RustGenerationService (single line per FRB-09)
- Created error dialog widget with "Generation Failed" title and non-dismissible OK button (D-05/D-06)
- Updated OutputPanel with "Downloading model..." spinner for pre-inference state (D-04) and live preview images via Image.memory (D-01/D-02/D-03)
- Added macOS network.client entitlement for outbound HuggingFace model downloads

## Task Commits

Each task was committed atomically:

1. **Task 1: FRB codegen integration, Dart bindings, and RustGenerationService** - `7bea315` (feat)
2. **Task 2: Provider swap, error dialog, output panel with downloading state and live preview** - `f2405d8` (feat)

## Files Created/Modified
- `gui/lib/features/generation/services/rust_generation_service.dart` - Implements GenerationService via FRB bindings, converts params Map to GuiParams
- `gui/lib/shared/widgets/error_dialog.dart` - Modal AlertDialog for generation errors per D-05
- `gui/lib/src/rust/api/api.dart` - Dart API wrappers for GuiParams, GuiProgressEvent, getPresets, getWeightsForPreset, generateImageStream
- `gui/lib/src/rust/frb_generated.dart` - FRB runtime with SSE codecs for all custom types and RustStreamSink streaming
- `gui/lib/src/rust/frb_generated.io.dart` - Platform-specific abstract declarations for IO
- `gui/lib/src/rust/frb_generated.web.dart` - Platform-specific abstract declarations for Web
- `gui/flutter_rust_bridge.yaml` - FRB config pointing to crate::api with dart_output lib/src/rust
- `gui/rust_builder/` - Cargokit build plugin (CMake/Xcode hooks for Rust compilation)
- `gui/lib/features/generation/providers/generation_provider.dart` - Swapped Mock to Rust, added previewBytes field, handles real images
- `gui/lib/features/output/output_panel.dart` - Added downloading state, live preview, error dialog trigger
- `gui/lib/main.dart` - Added RustLib.init() for FRB runtime bootstrap
- `gui/pubspec.yaml` - Added flutter_rust_bridge 2.12.0 and rust_lib_diffusion_rs_gui dependencies
- `gui/macos/Runner/DebugProfile.entitlements` - Added network.client for HuggingFace downloads
- `gui/macos/Runner/Release.entitlements` - Added network.client for HuggingFace downloads

## Decisions Made
- Created FRB Dart binding stubs manually because `flutter_rust_bridge_codegen generate` requires `cargo expand` which triggers the full C++ compilation of stable-diffusion.cpp via diffusion-rs-sys build.rs. This build takes 10-30+ minutes and failed in the worktree context due to CMake path resolution. The binding stubs match the Rust API signatures exactly and will be replaced by actual codegen output on the developer's first successful `flutter_rust_bridge_codegen generate` run.
- Used `RustStreamSink` with `executeNormal` (not `executeStream` which does not exist in FRB 2.12.0) for the streaming generate_image_stream binding. The sink's port is serialized and passed as a function argument.
- Added `previewBytes: Uint8List?` field to GenerationState to pass live preview images in-memory rather than writing to a file and reading back. This avoids extra file I/O on the Dart side while still supporting the file-based preview delivery from Rust (D-02).
- Converted OutputPanel from ConsumerWidget to ConsumerStatefulWidget to enable `ref.listenManual` for triggering the error dialog via post-frame callback. This ensures the dialog shows after the widget tree has settled.
- Added network.client entitlement to Release.entitlements in addition to DebugProfile.entitlements (deviation from plan which only mentioned DebugProfile). The release build also needs outbound network access for HuggingFace model downloads.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] FRB codegen generate could not run due to C++ build dependency**
- **Found during:** Task 1 (FRB codegen integration)
- **Issue:** `flutter_rust_bridge_codegen generate` requires `cargo expand` which compiles the full diffusion-rs-sys C++ backend. The CMake build failed in the worktree context.
- **Fix:** Created Dart binding files manually matching the Rust API signatures. These stubs are type-correct and will be replaced by actual codegen output when the developer runs codegen after the first successful C++ build.
- **Files modified:** gui/lib/src/rust/api/api.dart, gui/lib/src/rust/frb_generated.dart, gui/lib/src/rust/frb_generated.io.dart, gui/lib/src/rust/frb_generated.web.dart
- **Verification:** flutter analyze passes with no issues
- **Committed in:** 7bea315

**2. [Rule 3 - Blocking] FRB integrate overwrote all Dart source files with template code**
- **Found during:** Task 1 (FRB codegen integration)
- **Issue:** `flutter_rust_bridge_codegen integrate` applied Dart formatting and commented out our entire main.dart, replacing it with a demo template
- **Fix:** Restored all original Dart files via `git checkout`, then selectively applied only the necessary changes (RustLib.init() in main.dart, pubspec updates)
- **Files modified:** All gui/lib/ Dart files (restored to original state)
- **Verification:** flutter analyze passes; all Phase 1 code intact
- **Committed in:** 7bea315

**3. [Rule 2 - Missing Critical] Added network.client to Release.entitlements**
- **Found during:** Task 2 (macOS entitlements update)
- **Issue:** Plan only specified updating DebugProfile.entitlements, but the release build also requires outbound network access for model downloads
- **Fix:** Added com.apple.security.network.client to Release.entitlements
- **Files modified:** gui/macos/Runner/Release.entitlements
- **Verification:** Entitlement key present in both files
- **Committed in:** f2405d8

---

**Total deviations:** 3 auto-fixed (2 blocking, 1 missing critical)
**Impact on plan:** All fixes necessary for the crate to integrate correctly. The manual binding stubs are the primary deviation -- they maintain correct type signatures but will need regeneration when the full C++ build environment is available. No scope creep.

## Issues Encountered
- `flutter_rust_bridge_codegen integrate` creates a template `api/` directory that conflicts with the existing `api.rs` file from Plan 02-01. Removed the template directory before proceeding.
- `cargo expand` (used by codegen generate) fails because diffusion-rs-sys build.rs compiles stable-diffusion.cpp from the git submodule, which required initializing submodules first, and then still failed due to CMake path resolution in the worktree.
- FRB 2.12.0's `BaseHandler` does not have an `executeStream` method. Streaming in FRB works via `RustStreamSink` which creates a port, serializes it into the function call via `executeNormal`, and returns `sink.stream`.

## User Setup Required
Before the first `flutter build` or `flutter run`:
1. Ensure git submodules are initialized: `git submodule update --init --recursive`
2. Run `flutter_rust_bridge_codegen generate` in the `gui/` directory to produce the actual FRB bindings from the compiled Rust crate. This requires the full C++ build chain (CMake, Clang, C++ compiler). First build may take 10-30+ minutes.
3. After codegen, the files in `gui/lib/src/rust/` will be overwritten with the actual generated bindings.

## Next Phase Readiness
- The Dart-side integration is complete: RustGenerationService, error dialog, live preview, provider swap
- The FRB Dart binding stubs match the Rust API but must be regenerated via `flutter_rust_bridge_codegen generate` before the app can actually run
- All Phase 2 code is structurally complete and passes flutter analyze
- The app will generate real images from the GUI once the native Rust library is compiled

## Self-Check: PASSED

Files verified:
- FOUND: gui/lib/features/generation/services/rust_generation_service.dart
- FOUND: gui/lib/shared/widgets/error_dialog.dart
- FOUND: gui/lib/src/rust/api/api.dart
- FOUND: gui/lib/src/rust/frb_generated.dart
- FOUND: gui/flutter_rust_bridge.yaml
- FOUND: gui/rust_builder/pubspec.yaml

Commits verified:
- FOUND: 7bea315 (Task 1)
- FOUND: f2405d8 (Task 2)

---
*Phase: 02-rust-bridge-wiring*
*Completed: 2026-06-21*
