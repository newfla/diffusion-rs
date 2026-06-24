# Walking Skeleton -- diffusion-rs GUI

**Phase:** 1
**Generated:** 2026-06-18

## Capability Proven End-to-End

> The user can open a Flutter desktop app, see a two-panel layout with Yaru theming (light/dark/system toggle), press a Generate button, watch a mock progress bar advance over ~5 seconds, and see a placeholder image appear in the right panel.

## Architectural Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Framework | Flutter 3.x desktop (macOS, Linux, Windows) | Cross-platform desktop target matches the Rust backend; single codebase for all three OS |
| Design system | Yaru 10.x (`yaru` package) | Ubuntu-native look; provides light/dark themes, progress indicators, icons out of the box |
| State management | flutter_riverpod 3.x (Notifier / AsyncNotifier) | De facto standard; AsyncNotifier maps cleanly to generation lifecycle (idle/generating/complete/error) |
| Resizable panels | multi_split_view 3.x | Handles min/max constraints, drag cursor, hit testing; avoids 150+ lines of custom GestureDetector |
| File dialogs | file_picker 11.x | OS-native save dialog on all three platforms; saveFile() with default filename |
| Temp directory | path_provider 2.x + uuid 4.x | Platform-correct temp paths; UUID session isolation |
| Phase 1/2 seam | `GenerationService` abstract Dart class | MockGenerationService (Phase 1) swapped for RustGenerationService (Phase 2) via single provider line change |
| Directory layout | Feature-based folders under `gui/lib/` (params/, generation/, output/, shared/) per D-07 | Clear separation; each feature owns its providers, widgets, and services |
| Bridge crate location | `gui/rust/` as isolated Cargo workspace (not root workspace member) per D-08 / SETUP-02 | Avoids triggering CMake/GPU build on every cargo build in the monorepo |

## Stack Touched in Phase 1

- [x] Project scaffold (Flutter create, pubspec.yaml, analysis_options.yaml, platform runners)
- [x] Routing -- single-page app with two-panel layout (no router needed; state-driven content)
- [ ] Database -- N/A (no database in this desktop app; state is in-memory via Riverpod)
- [x] UI -- Generate button wired to mock service, progress bar updates, placeholder image display
- [x] Deployment -- `flutter run` on macOS/Linux/Windows exercises the full stack locally

Note: "Database" is not applicable to this Flutter desktop app. The equivalent data layer is the Riverpod provider state + temp filesystem, both exercised in the skeleton.

## Out of Scope (Deferred to Later Slices)

- Batch field (FORM-07) -- explicitly deferred per D-01
- Real Rust FFI integration -- Phase 2
- flutter_rust_bridge codegen -- Phase 2
- Image-to-image, ControlNet, LoRA UI -- out of v1 scope
- Model download progress UI -- v2
- Prompt history -- v2
- Gallery output panel -- v2

## Subsequent Slice Plan

Each later phase adds one vertical slice on top of this skeleton without altering its architectural decisions:

- Phase 2: Real Rust bridge wiring -- user can generate actual images via diffusion-rs FFI, with live preview updates and real presets from Rust
