---
phase: 02-rust-bridge-wiring
plan: 01
subsystem: ffi
tags: [flutter_rust_bridge, ffi, rust, streaming, preset, mpsc]

requires:
  - phase: 01-flutter-ui-foundation-mock-mode
    provides: GenerationService seam, ProgressEvent model, TempDirectoryManager

provides:
  - gui/rust/ Cargo crate with FRB-annotated functions
  - get_presets() returning all preset names as Vec<String>
  - get_weights_for_preset() returning weight variants per preset
  - generate_image_stream() with two-thread relay pattern and catch_unwind
  - GuiParams DTO with 17 FRB-compatible primitive fields
  - bridge module mapping GuiParams to diffusion-rs builders
  - Progress struct pub fields in src/api.rs

affects: [02-02, 02-03, dart-bindings, rust-generation-service]

tech-stack:
  added: [flutter_rust_bridge 2.12.0, anyhow 1.0]
  patterns: [StreamSink relay, catch_unwind defense-in-depth, GuiParams DTO mapping, frb_generated stub]

key-files:
  created:
    - gui/rust/Cargo.toml
    - gui/rust/src/lib.rs
    - gui/rust/src/api.rs
    - gui/rust/src/gui_params.rs
    - gui/rust/src/bridge.rs
    - gui/rust/src/frb_generated.rs
  modified:
    - src/api.rs

key-decisions:
  - "Empty [workspace] in gui/rust/Cargo.toml to isolate from root workspace (avoids adding to exclude list)"
  - "frb_generated.rs stub with placeholder StreamSink for pre-codegen compilation"
  - "Exhaustive match arms in map_preset and get_weights_for_preset (no catch-all) for compile-time safety on new presets"

patterns-established:
  - "StreamSink two-thread relay: worker thread calls gen_img_with_progress, relay thread bridges mpsc to StreamSink"
  - "GuiParams DTO pattern: primitives-only struct crosses FFI, mapped to builders inside Rust"
  - "Macro-based weight matching: with_weight! macro reduces boilerplate for preset-to-weight mapping"

requirements-completed: [FRB-01, FRB-02, FRB-03, FRB-04, FRB-05, FRB-06, FRB-07]

duration: 8min
completed: 2026-06-21
status: complete
---

# Phase 2 Plan 1: Rust Bridge Crate Summary

**gui/rust/ Cargo crate with three FRB functions (get_presets, get_weights_for_preset, generate_image_stream), GuiParams DTO, and Progress pub fields**

## Performance

- **Duration:** 8 min
- **Started:** 2026-06-21T17:38:07Z
- **Completed:** 2026-06-21T17:46:46Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments
- Created gui/rust/ as isolated Cargo crate with path dependency on diffusion-rs, passing cargo check
- Implemented get_presets() and get_weights_for_preset() with exhaustive match arms covering all 41 presets and 24 weight-bearing variants
- Implemented generate_image_stream() with two-thread relay pattern bridging mpsc::Receiver to StreamSink, catch_unwind defense, and file-based preview reading
- Created bridge::map_preset() with descriptive error messages (no panics) and build_configs() mapping all GUI parameters to PresetBuilder/ConfigBuilder/ModelConfigBuilder
- Made Progress struct fields pub in src/api.rs for FRB access

## Task Commits

Each task was committed atomically:

1. **Task 1: Scaffold gui/rust/ crate and make Progress fields pub** - `22ef0b3` (feat)
2. **Task 2: Implement FRB API functions and bridge mapping** - `2a6f74b` (feat)

## Files Created/Modified
- `gui/rust/Cargo.toml` - Isolated Cargo crate with diffusion-rs path dep, FRB, anyhow, strum; panic=abort release profile
- `gui/rust/src/lib.rs` - Module declarations and FRB init_app() function
- `gui/rust/src/api.rs` - Three FRB-annotated functions: get_presets, get_weights_for_preset, generate_image_stream
- `gui/rust/src/gui_params.rs` - GuiParams DTO with 17 FRB-compatible primitive fields
- `gui/rust/src/bridge.rs` - map_preset() and build_configs() mapping GuiParams to diffusion-rs builders
- `gui/rust/src/frb_generated.rs` - Placeholder StreamSink stub for pre-codegen compilation
- `src/api.rs` - Progress struct fields changed from private to pub

## Decisions Made
- Used empty `[workspace]` table in gui/rust/Cargo.toml to prevent Cargo from treating it as a child of the root workspace, rather than adding to the root's exclude list
- Created frb_generated.rs stub with placeholder StreamSink type so cargo check passes before FRB codegen runs
- Chose exhaustive match arms (no catch-all `_`) in map_preset and get_weights_for_preset so the compiler errors when new presets are added to diffusion-rs, forcing the bridge to be updated

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added empty [workspace] to gui/rust/Cargo.toml**
- **Found during:** Task 1 (Scaffold gui/rust/ crate)
- **Issue:** Cargo detected gui/rust/Cargo.toml as belonging to the root workspace because it was inside the repo tree
- **Fix:** Added `[workspace]` empty table to gui/rust/Cargo.toml to mark it as its own workspace root
- **Files modified:** gui/rust/Cargo.toml
- **Verification:** cargo check passes successfully
- **Committed in:** 22ef0b3

**2. [Rule 3 - Blocking] Initialized git submodules in worktree**
- **Found during:** Task 1 (cargo check verification)
- **Issue:** sys/stable-diffusion.cpp submodule was empty in the worktree, causing build.rs to fail finding C++ headers
- **Fix:** Ran `git submodule update --init --recursive`
- **Files modified:** None (submodule checkout)
- **Verification:** cargo check passes after submodule init
- **Committed in:** Not committed (submodule state is tracked by parent repo)

---

**Total deviations:** 2 auto-fixed (2 blocking)
**Impact on plan:** Both fixes necessary for the crate to compile. No scope creep.

## Issues Encountered
- First cargo check attempt failed because the worktree had empty submodule directories; resolved by initializing submodules
- Catch-all `_` patterns in match statements on PresetDiscriminants caused unreachable_patterns warnings; removed them since all variants are exhaustively matched

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- gui/rust/ crate compiles and is ready for FRB codegen integration (Plan 02-02)
- frb_generated.rs stub will be replaced by actual codegen output when flutter_rust_bridge_codegen runs
- All three FRB functions are implemented and ready for Dart binding generation
- Progress pub fields in src/api.rs enable FRB to access progress data

---
*Phase: 02-rust-bridge-wiring*
*Completed: 2026-06-21*
