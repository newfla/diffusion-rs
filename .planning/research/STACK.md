# Technology Stack — diffusion-rs GUI

**Project:** diffusion-rs GUI (Flutter desktop wrapping Rust image-generation library)
**Researched:** 2026-06-18
**Confidence:** MEDIUM (context7 unavailable; based on codebase inspection + training knowledge through August 2025; verify version pins before scaffolding)

---

## Recommended Stack

### Core Flutter / Dart

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| Flutter SDK | >=3.22.0 (stable channel) | UI framework | Minimum for desktop stability + Impeller on macOS/Linux; 3.22 ships Dart 3.4 |
| Dart SDK | >=3.4.0 | Language | Required by flutter_rust_bridge 2.x; records, patterns, sealed classes all available |

**Do NOT use Flutter beta/master channel.** Desktop rendering is production-stable on the stable channel since 3.x. Beta has broken desktop window management APIs multiple times.

**Minimum Dart SDK constraint in pubspec.yaml:**
```yaml
environment:
  sdk: ">=3.4.0 <4.0.0"
  flutter: ">=3.22.0"
```

---

### FFI Bridge

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| flutter_rust_bridge | ^2.7.0 | Dart ↔ Rust FFI code generation | Only mature, maintained solution for typed Dart/Rust FFI on desktop; 2.x rewrote codegen, is not compatible with 1.x |
| flutter_rust_bridge_codegen (Cargo) | =2.7.0 (match pub version) | Code generation binary | Runs `flutter_rust_bridge_codegen generate`; version must exactly match the pub package |
| cargokit (bundled) | (bundled by frb template) | Compiles Rust on Flutter build | Invoked automatically by the Flutter build system; no separate install |

**Version pinning is critical:** the pub package version and the `flutter_rust_bridge_codegen` Cargo binary version must be identical. A mismatch produces confusing code-generation errors. Pin both to the same patch release.

#### frb_codegen setup workflow

```bash
# 1. Install the codegen binary (pin to same version as pub dep)
cargo install flutter_rust_bridge_codegen --version 2.7.0 --locked

# 2. Create the Flutter project inside /gui (do NOT run flutter create inside
#    the existing Rust workspace root — it will collide with Cargo.toml)
cd /path/to/diffusion-rs
flutter create --template=app --platforms=macos,linux,windows gui
cd gui

# 3. Add flutter_rust_bridge to pubspec.yaml
flutter pub add flutter_rust_bridge

# 4. Add the frb crate to the Rust side (see Rust section below)
#    In gui/rust/Cargo.toml:
#    [dependencies]
#    flutter_rust_bridge = "2.7.0"

# 5. Write your bridge API in gui/rust/src/api/simple.rs
#    (annotate public functions with #[flutter_rust_bridge::frb])

# 6. Run codegen (from gui/ directory)
flutter_rust_bridge_codegen generate

# 7. Codegen outputs:
#    gui/lib/src/rust/frb_generated.dart       — Dart bindings
#    gui/rust/src/frb_generated.rs             — Rust glue (do not edit)
```

**Re-run codegen every time you change the Rust API.** Add it to your build scripts. The generated files are checked into git — they are large but necessary for the CI that does not have `flutter_rust_bridge_codegen` available.

#### Workspace integration

The Rust side for flutter_rust_bridge must live in a **separate Cargo workspace** from the main `diffusion-rs` workspace, or be added as a new workspace member. The recommended structure:

```
diffusion-rs/               ← existing workspace (members: sys, cli)
  Cargo.toml
  src/                      ← diffusion-rs lib crate
  gui/
    pubspec.yaml            ← Flutter project root
    lib/
    rust/                   ← NEW Cargo workspace for frb
      Cargo.toml            ← [workspace] with members = ["."]
      src/
        api/
          mod.rs
          generation.rs     ← bridge functions
        lib.rs
```

The `gui/rust/Cargo.toml` workspace can add `diffusion-rs` as a path dependency:

```toml
[package]
name = "diffusion_rs_gui"
version = "0.1.0"
edition = "2024"

[dependencies]
flutter_rust_bridge = "2.7.0"
diffusion-rs = { path = "../.." }   # path to the root lib crate

[lib]
crate-type = ["cdylib"]
```

**Do NOT add `cdylib` to the main `diffusion-rs` crate.** Keep the frb crate as a thin adapter layer that imports from `diffusion-rs` and exposes only what the GUI needs.

---

### Design System

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| yaru | ^6.1.0 | Design system (Ubuntu/GNOME look) | Required by project spec; looks coherent on Linux and acceptable on macOS/Windows; provides both light and dark themes out of the box |
| yaru_icons | ^2.3.0 | Icon set matching Yaru design | Always use this alongside yaru; mixing with Material icons breaks visual coherence |

**Version note:** Yaru 6.x requires Flutter >=3.19. At Flutter 3.22+ everything is compatible. Do NOT use yaru 3.x or 4.x — they have breaking theme API differences.

#### Theme setup

```dart
// main.dart
import 'package:flutter/material.dart';
import 'package:yaru/yaru.dart';

void main() async {
  await YaruWindowTitleBar.ensureInitialized(); // required before runApp on desktop
  runApp(const DiffusionApp());
}

class DiffusionApp extends StatelessWidget {
  const DiffusionApp({super.key});

  @override
  Widget build(BuildContext context) {
    return YaruTheme(
      builder: (context, yaru, child) {
        return MaterialApp(
          theme: yaru.theme,           // light
          darkTheme: yaru.darkTheme,   // dark
          // themeMode defaults to ThemeMode.system (follows OS setting)
          // Override for manual toggle:
          // themeMode: _isDark ? ThemeMode.dark : ThemeMode.light,
          home: child,
        );
      },
      child: const MainPage(),
    );
  }
}
```

**Manual light/dark toggle pattern:**

```dart
// Use a ValueNotifier at the app level, passed down via InheritedWidget or provider
final _themeModeNotifier = ValueNotifier(ThemeMode.system);

// In build:
themeMode: _themeModeNotifier.value,

// Toggle button:
IconButton(
  icon: Icon(
    _themeModeNotifier.value == ThemeMode.dark
        ? YaruIcons.weather_clear_night
        : YaruIcons.weather_clear,
  ),
  onPressed: () {
    _themeModeNotifier.value =
        _themeModeNotifier.value == ThemeMode.dark
            ? ThemeMode.light
            : ThemeMode.dark;
  },
)
```

#### Key Yaru widgets for this project

| Widget | Use in this project |
|--------|---------------------|
| `YaruWindowTitleBar` | Custom title bar with integrated controls (replaces default OS chrome on Linux/Windows) |
| `YaruToggleButton` / `YaruCheckButton` | Low VRAM toggle, preview mode, upscaler toggle |
| `YaruProgressBar` | Generation progress |
| `YaruSection` / `YaruTile` | Grouping parameters in the left panel |
| `YaruPasswordField` | HuggingFace token with built-in toggle visibility — use this directly instead of rolling your own |
| `YaruAutocomplete` | (optional) prompt history |
| `YaruBanner` | Error/success notifications |
| `YaruDialogTitleBar` | Dialogs (e.g., save confirmation) |

**Do NOT use `YaruWindowTitleBar` on macOS** — macOS has its own native title bar and adding a Flutter-drawn one creates a double title bar. Conditionally omit it:

```dart
if (!Platform.isMacOS) const YaruWindowTitleBar(),
```

---

### Desktop Window Management

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| window_manager | ^0.4.0 | Programmatic window size, position, min-size | Required to set minimum window size (prevent panel collapse) and handle close event for temp file cleanup |
| bitsdojo_window | — | NOT recommended | Abandoned; use window_manager instead |

#### Setup

Add to `linux/`, `macos/`, `windows/` entrypoints — `window_manager` requires the standard one-time setup call:

```dart
// main.dart
import 'package:window_manager/window_manager.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await windowManager.ensureInitialized();

  WindowOptions windowOptions = const WindowOptions(
    size: Size(1280, 800),
    minimumSize: Size(900, 600),
    center: true,
    title: 'diffusion-rs',
  );
  windowManager.waitUntilReadyToShow(windowOptions, () async {
    await windowManager.show();
    await windowManager.focus();
  });

  await YaruWindowTitleBar.ensureInitialized();
  runApp(const DiffusionApp());
}
```

#### Close event for temp file cleanup

```dart
class _MainPageState extends State<MainPage> with WindowListener {
  @override
  void initState() {
    super.initState();
    windowManager.addListener(this);
    windowManager.setPreventClose(true); // intercept close
  }

  @override
  void onWindowClose() async {
    await _cleanupTempFiles();     // delete temp dir
    await windowManager.destroy(); // actually close
  }
}
```

**macOS note:** `NSApplicationSupportsSecureRestorableState` in `Info.plist` must be set to `YES` to suppress a console warning on macOS 14+.

**Linux note:** The `window_manager` package requires GTK 3.x headers. On Ubuntu this means `libgtk-3-dev`. This is already present on most dev systems but needs to be declared in your CI Dockerfile.

---

### File System / Temp Files

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| path_provider | ^2.1.4 | Get OS temp dir, app documents dir | Flutter-first, platform-verified, no native code to maintain |

#### Temp directory pattern

```dart
import 'package:path_provider/path_provider.dart';
import 'dart:io';

class TempDirManager {
  late Directory _sessionDir;

  Future<void> init() async {
    final base = await getTemporaryDirectory();
    _sessionDir = await Directory(
      '${base.path}/diffusion_rs_${DateTime.now().millisecondsSinceEpoch}',
    ).create(recursive: true);
  }

  String get previewPath => '${_sessionDir.path}/preview.png';
  String get outputPath  => '${_sessionDir.path}/output.png';

  Future<void> cleanup() async {
    if (await _sessionDir.exists()) {
      await _sessionDir.delete(recursive: true);
    }
  }
}
```

`getTemporaryDirectory()` returns:
- macOS: `$TMPDIR` (app-scoped, cleaned by OS on reboot)
- Linux: `/tmp`
- Windows: `%TEMP%` or `%TMP%`

**Do NOT use `getApplicationDocumentsDirectory()` for temp output.** It is user-visible and persists. Use it only if the user explicitly clicks "Save" to copy the final image out of temp.

---

### Image Display

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| (built-in Flutter) | — | `Image.file` / `Image.memory` | No dependency needed; both ship with Flutter |

#### Recommended pattern: `Image.file` with cache-busting

For preview images that update on disk periodically, use `Image.file` with a `key` that changes to force widget rebuild:

```dart
// State variable updated whenever preview is refreshed
int _previewVersion = 0;

Image.file(
  File(tempDirManager.previewPath),
  key: ValueKey(_previewVersion), // changing key forces image reload
  fit: BoxFit.contain,
  errorBuilder: (ctx, err, stack) =>
      const Center(child: Icon(YaruIcons.image_missing_symbolic)),
)

// Called when Rust reports a new preview written to disk:
setState(() => _previewVersion++);
```

**Do NOT use `Image.file` without a changing `key`**. Flutter caches image data by path. If the file at the path changes but the key does not, Flutter displays the stale cached version.

**Alternatively, use `Image.memory` (Uint8List)**. If the Rust bridge returns image bytes directly instead of writing to disk, `Image.memory` avoids the file I/O round-trip and the cache-busting problem entirely. For previews during generation this is often cleaner:

```dart
// frb bridge returns Uint8List from Rust
final Uint8List previewBytes = await api.getPreviewBytes();
Image.memory(previewBytes, fit: BoxFit.contain)
```

For Phase 1 (mock), use `Image.asset` with a placeholder PNG bundled in `assets/`.

---

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `provider` | ^6.1.2 | State management | App-level state (theme mode, generation status); simple enough that Riverpod is overkill for this project |
| `file_picker` | ^8.1.2 | Output folder picker | User chooses where to save final image; avoid rolling native dialogs |
| `intl` | ^0.19.0 | Number formatting | Seed field, step count — format large integers |
| `collection` | ^1.19.0 | List/map utilities | Preset → weight options mapping |

**Do NOT add riverpod or bloc** for Phase 1. The state is a single generation form — a simple `ChangeNotifier` with `provider` covers it. Revisit only if Phase 2 wiring adds complexity (stream management for progress, cancellation tokens).

---

## Rust Side for flutter_rust_bridge

### What to expose

Only expose a thin, GUI-specific API from the frb crate. The existing `diffusion-rs` library uses `unsafe` FFI extensively and carries types (`CLibString`, `CLibPath`, raw pointers) that cannot cross the frb boundary.

The frb bridge crate (`gui/rust/src/api/generation.rs`) should define a clean DTO layer:

```rust
// gui/rust/src/api/generation.rs
use flutter_rust_bridge::frb;

/// All parameters the GUI needs to pass for a generation run.
/// Keep types frb-compatible: String, i32, i64, f32, bool, Option<T>.
#[frb(dart_metadata=("freezed"))]  // optional: generate Dart Freezed class
pub struct GenerationParams {
    pub preset: String,          // PresetDiscriminants variant name
    pub weights: Option<String>, // WeightType variant name, if applicable
    pub prompt: String,
    pub negative: Option<String>,
    pub steps: Option<i32>,
    pub width: Option<i32>,
    pub height: Option<i32>,
    pub batch: i32,
    pub output_path: String,     // temp dir path provided by Dart
    pub preview_path: String,    // temp dir preview path
    pub cache_mode: Option<String>,
    pub preview_mode: Option<String>,
    pub upscaler: Option<String>,
    pub upscaler_scale: f32,
    pub token: Option<String>,
    pub low_vram: bool,
    pub seed: i64,
}

/// Result type — simple enough to cross the boundary
pub struct GenerationResult {
    pub output_path: String,
    pub elapsed_ms: u64,
}

/// Called from Dart. Runs synchronously on a background thread managed by frb.
/// Use #[frb(sync)] only for trivially-fast calls (e.g., list-presets).
/// For generation, use the default (async via frb's Rust thread pool).
pub async fn generate_image(params: GenerationParams) -> anyhow::Result<GenerationResult> {
    // parse params, build Config + ModelConfig, call diffusion_rs::api::gen_img
    todo!()
}

/// Returns the list of preset discriminant names for the dropdown.
/// This is safe to make sync because it does no I/O.
#[frb(sync)]
pub fn list_presets() -> Vec<String> {
    use diffusion_rs::preset::PresetDiscriminants;
    use strum::VariantNames;
    PresetDiscriminants::VARIANTS.iter().map(|s| s.to_string()).collect()
}

/// Returns weight options valid for a given preset name.
#[frb(sync)]
pub fn list_weights_for_preset(preset: String) -> Vec<String> {
    // match on preset name, return applicable WeightType variants
    todo!()
}
```

### Async patterns

- **Default (async):** frb 2.x runs Rust futures on its own thread pool. `generate_image` should be `async fn` even if the body is synchronous-blocking — frb spawns it off the main thread automatically.
- **Progress reporting:** Use `flutter_rust_bridge::StreamSink<ProgressUpdate>` to stream step-by-step progress to Dart. This replaces the `mpsc::Sender<Progress>` used internally by `gen_img_with_progress`.
- **Do NOT expose `ModelConfig` or `Config` directly across the boundary.** They contain raw pointers and non-`Send` types. The bridge function must own, build, and drop them inside the Rust call.

```rust
pub struct ProgressUpdate {
    pub step: i32,
    pub total_steps: i32,
}

pub async fn generate_with_progress(
    params: GenerationParams,
    sink: StreamSink<ProgressUpdate>,
) -> anyhow::Result<GenerationResult> {
    // Use std::sync::mpsc internally, relay to sink
    let (tx, rx) = std::sync::mpsc::channel::<diffusion_rs::api::Progress>();
    // ... spawn thread, call gen_img_with_progress, relay rx messages to sink
    todo!()
}
```

### What NOT to expose

- Do not expose `ModelConfig`, `Config`, `CLibString`, `CLibPath` — they are not frb-compatible.
- Do not expose `DiffusionError` directly; wrap it in `anyhow::Error` and let frb convert it to a Dart exception.
- Do not make `generate_image` `#[frb(sync)]` — it blocks for minutes; calling it synchronously would freeze the Dart UI.

---

## Platform-Specific Considerations

### macOS

- Add `com.apple.security.cs.allow-jit` entitlement if needed for JIT (usually not needed for Flutter desktop).
- `NSDocumentsFolderUsageDescription` in `Info.plist` if using file picker to write output.
- Metal acceleration is available in `diffusion-rs` via `--features metal`; expose as a conditional compile flag in the frb crate (`#[cfg(feature = "metal")]`).
- The macOS bundle requires code signing for distribution; for local dev, disable hardened runtime or use an ad-hoc signature.
- `YaruWindowTitleBar` — **omit on macOS**, as noted above.

### Linux

- GTK 3.x headers required for `window_manager` compilation.
- Vulkan is the GPU backend on Linux (`--features vulkan`).
- Test on both X11 and Wayland. `window_manager` 0.4.x works on both but the `setPreventClose` API may behave differently on Wayland — verify cleanup path.
- File picker on Linux uses `zenity` or `kdialog` depending on the desktop; `file_picker` handles this transparently.

### Windows

- CUDA is the GPU backend (`--features cuda`).
- The Rust DLL must be in the same directory as the Flutter executable or on `PATH`. The cargokit bundler handles this for debug builds; for release, verify the bundle output.
- Windows Defender sometimes false-positives on newly compiled DLLs; this is a user-facing issue, not a code issue.
- Use `window_manager` 0.4.x — the 0.3.x series has a known issue where `setPreventClose` does not fire on Alt+F4 on Windows 11.

---

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| FFI bridge | flutter_rust_bridge 2.x | uniffi-bindgen-dart | No maintained Flutter integration; frb is the standard |
| FFI bridge | flutter_rust_bridge 2.x | dart:ffi raw | Would require hand-writing all bindings; maintenance nightmare given the API surface |
| Design system | yaru ^6.x | Material (default Flutter) | Project requires Yaru per spec; Material does not give the GNOME aesthetic |
| Design system | yaru ^6.x | fluent_ui | Windows-only aesthetic; breaks on Linux/macOS |
| State management | provider | riverpod | Overkill for a single-form app in Phase 1; add if Phase 2 complexity justifies it |
| State management | provider | bloc | Too much boilerplate for this scope |
| Window management | window_manager | bitsdojo_window | Abandoned since 2022, no null-safety migration finished |
| Image display | Image.file + key | extended_image | Unnecessary dependency; built-in handles all required use cases |
| Temp files | path_provider | hardcoded /tmp | Platform-incorrect (Windows uses %TEMP%, macOS uses app-scoped TMPDIR) |

---

## Installation

```bash
# In gui/ (Flutter project root)
flutter pub add flutter_rust_bridge
flutter pub add yaru yaru_icons
flutter pub add window_manager
flutter pub add path_provider
flutter pub add provider
flutter pub add file_picker

flutter pub add --dev build_runner  # if using code generation for provider/etc

# Rust tooling
cargo install flutter_rust_bridge_codegen --version 2.7.0 --locked
```

**pubspec.yaml snippet (pinned):**
```yaml
dependencies:
  flutter:
    sdk: flutter
  flutter_rust_bridge: "^2.7.0"
  yaru: "^6.1.0"
  yaru_icons: "^2.3.0"
  window_manager: "^0.4.0"
  path_provider: "^2.1.4"
  provider: "^6.1.2"
  file_picker: "^8.1.2"

environment:
  sdk: ">=3.4.0 <4.0.0"
  flutter: ">=3.22.0"
```

---

## Confidence Assessment

| Area | Confidence | Reason |
|------|-----------|--------|
| flutter_rust_bridge 2.x setup | MEDIUM | frb 2.x released and stable as of mid-2024; workspace layout and codegen workflow are well-documented in frb docs; specific version 2.7.0 is a reasonable pin but verify latest before scaffolding |
| Yaru package API | MEDIUM | Yaru 6.x API shape (YaruTheme builder pattern, YaruPasswordField) is stable; version 6.1.x confirmed active; cross-check CHANGELOG before scaffolding |
| Flutter desktop platform handling | MEDIUM | window_manager 0.4.x, path_provider 2.x, Image.file cache-busting are established patterns |
| Rust bridge API design | HIGH | Based on direct codebase inspection of diffusion-rs types; the DTO boundary is clearly necessary given raw pointer types in ModelConfig/Config |

---

## Sources

- Codebase inspection: `/Users/flavio.bizzarri/repo/diffusion-rs/src/api.rs`, `cli/src/main.rs`, `Cargo.toml` (HIGH — first-party)
- flutter_rust_bridge documentation: https://cjycode.com/flutter_rust_bridge/ (MEDIUM — training knowledge, verify current)
- Yaru Flutter package: https://pub.dev/packages/yaru (MEDIUM — training knowledge, verify current)
- window_manager package: https://pub.dev/packages/window_manager (MEDIUM)
- path_provider package: https://pub.dev/packages/path_provider (MEDIUM)
