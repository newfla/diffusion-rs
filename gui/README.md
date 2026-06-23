# diffusion-rs GUI

Desktop GUI for diffusion-rs that exposes all CLI options in a two-panel interface (left: generation parameters, right: preview and final image). Communicates with the Rust library via flutter_rust_bridge (FFI).

## Prerequisites

- **Flutter SDK >= 3.32.x** (Dart SDK ^3.12.1 bundled) — install via [flutter.dev/docs/get-started/install](https://flutter.dev/docs/get-started/install) or `fvm`
- **Rust toolchain (stable, Edition 2024)** — install via [rustup.rs](https://rustup.rs); run `rustup default stable` after installing
- **Cargo** — bundled with the Rust toolchain (no separate install needed)
- **macOS: Xcode with Command Line Tools** — required for Metal and Accelerate frameworks and CocoaPods; install via `xcode-select --install`
- **CocoaPods** — install via `sudo gem install cocoapods`
- **CMake >= 3.15 and a C++ compiler** — required by the stable-diffusion.cpp submodule compiled inside `gui/rust/`; on macOS the Xcode Command Line Tools supply both

> Note: The full C++ backend (stable-diffusion.cpp) is compiled when the Rust crate is built for the first time. This requires the same native toolchain as building diffusion-rs from source (Clang, CMake, and the selected GPU SDK — Metal on macOS). The first build may take several minutes.

## Build and Run (macOS)

1. Clone the repo and initialise submodules:
   ```sh
   git clone --recurse-submodules <repo-url>
   ```

2. Enter the GUI directory:
   ```sh
   cd gui
   ```

3. Install Flutter dependencies:
   ```sh
   flutter pub get
   ```

4. Install CocoaPods dependencies:
   ```sh
   cd macos && pod install && cd ..
   ```

5. Run in debug mode (Cargokit builds the Rust crate automatically):
   ```sh
   flutter run -d macos
   ```

6. Build a release app bundle:
   ```sh
   flutter build macos --release
   ```

## FRB Codegen Caveat

flutter_rust_bridge 2.x generates the Dart bindings in `gui/lib/src/rust/` from the public API declared in `gui/rust/src/api.rs`.

Regenerating the bindings requires the full Rust + C++ build to succeed. To regenerate, run from the `gui/` directory:

```sh
flutter_rust_bridge_codegen generate
```

Pre-generated bindings are checked in to the repository. **You do not need to regenerate them unless you change `gui/rust/src/api.rs`.** Changing only Dart or Flutter code never requires regeneration.

The `gui/rust/` crate is intentionally **not** a member of the root workspace `Cargo.toml`. This prevents a plain `cargo build` at the repository root from triggering the expensive CMake / GPU backend compilation that belongs to the GUI build only.

## Project Structure

| Path | Purpose |
|---|---|
| `gui/lib/` | Dart/Flutter application code |
| `gui/lib/features/` | Feature modules: generation, output, params, models, services, theme, widgets |
| `gui/lib/shared/` | Shared Dart utilities and cross-feature components |
| `gui/lib/src/rust/` | FRB-generated Dart bindings — do not edit by hand |
| `gui/rust/` | Rust crate (`rust_lib_diffusion_rs_gui`) compiled by Cargokit |
| `gui/rust/src/api.rs` | Public API exposed to Dart via FRB |
| `gui/rust_builder/` | Cargokit integration package (`rust_lib_diffusion_rs_gui` Flutter plugin) |
| `gui/assets/` | Static assets (placeholder.png, etc.) |
| `gui/macos/` | macOS platform runner shell |
| `gui/linux/` | Linux platform runner shell |
| `gui/windows/` | Windows platform runner shell |

## Key Dependencies

| Package | Version | Purpose |
|---|---|---|
| flutter_rust_bridge | 2.12.0 | Dart to Rust FFI code generation |
| yaru | ^10.2.0 | Ubuntu/GNOME-style design system |
| flutter_riverpod | ^3.3.2 | State management (AsyncNotifier pattern) |
| multi_split_view | ^3.6.2 | Resizable two-panel layout |
| file_picker | ^11.0.2 | Native file and directory picker dialogs |
| path_provider | ^2.1.6 | Temp dir for output images |
