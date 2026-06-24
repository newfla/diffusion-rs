# Domain Pitfalls

**Domain:** Flutter desktop + flutter_rust_bridge + Rust image-generation library
**Researched:** 2026-06-18
**Overall confidence:** MEDIUM (flutter_rust_bridge v2 docs available via training knowledge; Yaru cross-platform behavior verified; Rust FFI pitfalls from codebase analysis)

---

## Critical Pitfalls

Mistakes that cause rewrites, data loss, or crashes.

---

### Pitfall C-1: Calling Synchronous Rust Bridge Functions from the Main Isolate

**What goes wrong:** flutter_rust_bridge v2 distinguishes sync from async Rust functions at the API boundary. If the Rust function is declared as a plain (non-async) function, the bridge calls it synchronously on the calling Dart isolate. For a generation that takes 30–300 seconds, this completely freezes the Flutter UI — no repaints, no button response, no progress bar updates — until the Rust function returns. The UI is dead, not just slow.

**Why it happens:** Developers new to flutter_rust_bridge assume "the bridge handles threading." It does for `async` Rust functions, which are dispatched to a Dart isolate automatically. But sync Rust functions are not. The diffusion-rs API (`generate()`, `upscale()`) are synchronous blocking calls.

**Consequences:** App appears frozen/crashed. On macOS, the OS may show the spinning beach ball and offer to force-quit. On Windows, the window stops responding and gets a "not responding" title. Users lose trust.

**Prevention:**
- Wrap every long-running Rust call in a `Future.run()` / `compute()` or ensure the bridge function is declared `async` on the Dart side.
- With flutter_rust_bridge v2, annotate long-running Rust functions with `#[frb(sync = false)]` (the default for `async fn`) or expose them as `async fn` in Rust using Tokio/blocking spawn internally.
- Verify in Phase 2 (wiring) by adding a timer widget — if it stops ticking, the main isolate is blocked.

**Warning signs:**
- Progress bar widget stops animating during generation.
- `flutter run` console shows "I/flutter: Skipped N frames" or ANR-style warnings.
- The Start button cannot be clicked again even to cancel.

**Phase that must address it:** Phase 2 (FFI wiring). Phase 1 (mock) is safe because mock progress uses `Future.delayed`.

---

### Pitfall C-2: Rust Panic Propagating Across the FFI Boundary

**What goes wrong:** Rust panics that cross an `extern "C"` FFI boundary are **undefined behavior**. With the existing codebase's 40+ `.unwrap()` calls (see CONCERNS.md), any unexpected input — non-UTF-8 path, poisoned `RwLock`, missing LoRA file — can trigger a panic inside the C-callable function. In Rust before edition 2024 stable with `-C panic=abort`, the default is `panic=unwind`, but unwinding across FFI is UB and will corrupt the Dart VM heap, causing a silent crash with no error message in the Flutter UI.

**Why it happens:** The diffusion-rs library was designed as a CLI tool. Panics are acceptable in CLI context (process exits cleanly). Via FFI, the same panics become heap corruption.

**Consequences:** App crashes silently with no error shown to the user. The crash may be non-deterministic (depends on whether the panic unwinds through the FFI boundary or aborts). On macOS/Linux, this manifests as SIGABRT or SIGSEGV with a cryptic backtrace. On Windows, it is an access violation.

**Prevention:**
- Compile the Rust library with `panic = "abort"` in `[profile.release]` of the GUI crate's `Cargo.toml`. This converts panics to clean process aborts rather than UB heap corruption. The app will still crash, but predictably.
- Wrap every top-level FFI-exposed function in `std::panic::catch_unwind()` and return an error code instead of panicking.
- In Phase 2, add a `catch_unwind` wrapper around the flutter_rust_bridge entry points that calls the diffusion-rs API. The bridge itself cannot protect you from panics inside your Rust code.
- Long-term: fix the underlying `.unwrap()` calls in diffusion-rs (tracked in CONCERNS.md).

**Warning signs:**
- App disappears without an error dialog when attempting generation.
- Console shows `signal: 6, SIGABRT` or similar.
- Inconsistent crashes (sometimes works, sometimes not) — hallmark of UB.

**Phase that must address it:** Phase 2 (FFI wiring) — must add `catch_unwind` wrappers before any user testing.

---

### Pitfall C-3: Progress Callback Threading — Calling Dart from a Non-Dart Thread

**What goes wrong:** The diffusion-rs progress and preview callbacks (`unsafe extern "C"` at `src/api.rs:1513` and `src/api.rs:1540`) fire from a C++ thread that is NOT the Dart main isolate thread. When flutter_rust_bridge forwards these as `StreamSink` events, it must post to the Dart event loop. If the `StreamSink` is used incorrectly (e.g., accessed from the wrong thread, or dropped before the stream is consumed), the app crashes or the callback silently fires into the void.

**Why it happens:** Developers assume the callback is called from the Rust async runtime, which the bridge manages. But these callbacks come from C++ (stable-diffusion.cpp's thread pool) via the `unsafe extern "C"` boundary — they are on a raw OS thread with no Dart VM attachment.

**Consequences:**
- Crash with "Bad state: Stream has already been listened to" or "Null check operator used on a null value."
- Progress bar never updates (events lost).
- Dart heap corruption if the StreamSink is used after the Dart isolate has been torn down.

**Prevention:**
- Use `StreamSink<T>` from flutter_rust_bridge v2 — it is thread-safe by design and queues events to the correct Dart port.
- Never store a raw Dart port or callback pointer in C++ code; always go through the bridge's managed channel.
- On the Dart side, always `cancel()` the stream subscription before disposing the widget, not after.
- Set up stream early (before calling generate) so no events are lost in the window between function call and listener attachment.

**Warning signs:**
- Progress events arrive in batches (buffer filling up) rather than smoothly.
- Debug console shows "Bad state" or "Broken pipe" when stream is done.
- Preview image never updates despite progress firing.

**Phase that must address it:** Phase 2 (FFI wiring). In Phase 1, simulate with a `Stream.periodic` to validate the UI wiring pattern before real callbacks.

---

### Pitfall C-4: Codegen Out of Sync — Generated Bridge Code Not Matching Rust API

**What goes wrong:** flutter_rust_bridge v2 generates Dart binding code (`frb_generated.dart`) and Rust glue code (`frb_generated.rs`) from the Rust source. If the Rust API changes (function signature, new type, removed function) but the codegen is not re-run, the mismatch causes one of: (a) compile error in Rust, (b) `MissingPluginException` or symbol-not-found at runtime, or (c) type mismatch panic in the bridge layer.

**Why it happens:** The codegen step is an explicit manual step (`flutter_rust_bridge_codegen generate`) that is easy to forget. In a monorepo where developers edit the Rust library and the Flutter GUI in the same commit, the codegen is often the forgotten third step.

**Consequences:** The app builds but crashes at first bridge call. Or worse — it silently uses stale generated code and calls the wrong function signature.

**Prevention:**
- Add codegen to the Rust library's `build.rs` so `cargo build` in the GUI crate re-runs codegen automatically. (`flutter_rust_bridge_codegen_build` crate provides this.)
- Add a CI check: run codegen, then `git diff --exit-code` on the generated files. If they drift, the CI fails.
- In the monorepo, document the exact build sequence: `cargo build -p diffusion-rs-gui` triggers codegen; `flutter pub get` + `flutter build` follows.

**Warning signs:**
- `flutter run` succeeds but first bridge call throws `MissingPluginException`.
- Error message: "symbol not found: frb_dart_fn_deliver_output".
- Compilation succeeds but function arity mismatches at runtime.

**Phase that must address it:** Phase 2 (FFI wiring) setup — the very first task is establishing the codegen pipeline reliably before writing any bridge logic.

---

## Moderate Pitfalls

---

### Pitfall M-1: Cargo Workspace + Flutter Subfolder — Linking and Artifact Location

**What goes wrong:** When the Flutter app lives in `/gui` inside the diffusion-rs Cargo workspace, `flutter build` does not know where `cargo build` put the compiled `.dylib`/`.so`/`.dll`. The flutter_rust_bridge template assumes the Rust crate IS the Flutter project root's `rust/` subdirectory. When the structure diverges (workspace root above Flutter root), the `cargoKit` or manual build script must be explicitly told the artifact path.

**Why it happens:** flutter_rust_bridge v2 uses `cargokit` for the native build integration. `cargokit` reads `pubspec.yaml` to find the Rust crate. If the Flutter `pubspec.yaml` is at `/gui/pubspec.yaml` but the Cargo workspace is at `/Cargo.toml`, the relative path `../` must be correctly specified in `pubspec.yaml`'s `flutter_rust_bridge` section.

**Consequences:**
- `flutter build` succeeds but does not embed the native library — app starts, then crashes with "Failed to load dynamic library."
- On macOS, the `.dylib` ends up in `target/` but is not copied to `macos/Runner/Frameworks/`.
- On Windows, the `.dll` is not placed alongside the `.exe`.

**Prevention:**
- Set `crate-type = ["cdylib", "staticlib"]` in the GUI Rust lib's `Cargo.toml`.
- Configure the `pubspec.yaml` `flutter_rust_bridge` section with explicit `crate_dir: ../` (path from Flutter root to Rust crate).
- Verify with `otool -L` (macOS) or `dumpbin /dependents` (Windows) that the final bundle links the correct library.
- Add a smoke test to CI: build the release Flutter app, verify the native library is present in the bundle.

**Warning signs:**
- `flutter run --debug` works (uses debug dylib in `target/debug/`) but `flutter run --release` fails.
- Error: "Failed to load dynamic library: libdiffusion_rs_gui.dylib: image not found."
- Linux: `error while loading shared libraries: libdiffusion_rs_gui.so`.

**Phase that must address it:** Phase 2 (FFI wiring) — environment setup section.

---

### Pitfall M-2: No Cancellation Path — Generation Cannot Be Stopped

**What goes wrong:** stable-diffusion.cpp does not expose a cancellation API. Once generation starts, it runs to completion (or crash). The Flutter UI provides a "Stop" button but has no way to honour it. If the Dart isolate is killed while the Rust/C++ code is executing, the C++ context is not freed, causing a memory leak and potentially leaving GPU resources locked until process exit.

**Why it happens:** Cancellation is rarely designed in at the start. The Rust API wraps the C++ library which has no abort signal. The common workaround (killing the Dart isolate) does not stop the native thread.

**Consequences:**
- Users wait for the full generation even after clicking Stop.
- Repeated "stops" without actually stopping accumulate leaked C++ contexts.
- On VRAM-limited systems, leaked contexts prevent the next generation from starting.

**Prevention:**
- In Phase 1 (mock), design the Stop button so it clearly communicates "will stop after current step" rather than "stops immediately" — set user expectations.
- In Phase 2, implement a Rust-side atomic bool abort flag. Pass it into the generation loop. Check it in the progress callback. This requires wrapping the C++ call or using a thread that can be polled.
- Alternative: use `process::exit()` as a last resort (nuclear option) — acceptable for desktop apps where the user explicitly requests it.
- Do not implement a "kill the isolate" cancellation — it does not stop native threads and leaks resources.

**Warning signs:**
- Stop button is wired to `isolate.kill()` — this is the anti-pattern.
- Progress bar continues advancing after Stop is pressed.
- Second generation attempt fails with "model already loaded" or VRAM error.

**Phase that must address it:** Phase 1 — Stop button design (communicates pending stop); Phase 2 — actual abort mechanism.

---

### Pitfall M-3: token.txt Compile-Time Dependency Breaking Fresh Builds

**What goes wrong:** `src/preset.rs` and `src/modifier.rs` use `include_str!("../token.txt")` evaluated at compile time. When the GUI Rust crate depends on the diffusion-rs library crate, `cargo build` for the GUI triggers a full library recompile — including the test files that contain `include_str!`. Even though those tests are `#[ignore]`d, the `include_str!` macro still runs at compile time. If `token.txt` does not exist, the build fails with a confusing "file not found" error that has nothing to do with the GUI.

**Why it happens:** The `include_str!` is in test code, and Rust compiles test code even when tests are not run if the `#[cfg(test)]` attribute is present in a dependency.

**Consequences:**
- CI/CD for the GUI fails on fresh checkout with: `error: couldn't read ../token.txt: No such file or directory`.
- New developer onboarding hits this immediately.
- If `token.txt` contains a real HuggingFace token and is committed to git (currently visible in root per CONCERNS.md), it is exposed in git history.

**Prevention:**
- Create a placeholder empty `token.txt` in the repo root (no real token) with a clear comment.
- Add `token.txt` to `.gitignore` and document that users must create it locally with their real token only if needed for testing.
- In the GUI's `Cargo.toml`, depend on the diffusion-rs library with `default-features = false` and exclude test features.
- Guard the `include_str!` calls with `#[cfg(test)]` only in files that need it, so they are only compiled during `cargo test`, not `cargo build`. (This is a fix that should be contributed back to diffusion-rs.)

**Warning signs:**
- Fresh checkout fails at the Rust compile step with `couldn't read ../token.txt`.
- The root directory contains a `token.txt` file tracked in git.

**Phase that must address it:** Phase 2 setup — document the workaround clearly in the GUI's README; Phase 2 execution — create the placeholder file.

---

### Pitfall M-4: Windows Path Handling — Non-UTF-8 Paths Silently Dropped

**What goes wrong:** The diffusion-rs library uses `to_str().unwrap_or_default()` in `SafePathBuf` conversions (CONCERNS.md, `src/api.rs:1291`). On Windows, user profile paths can contain characters outside UTF-8 (e.g., certain CJK usernames). The `unwrap_or_default()` silently converts such paths to empty strings, causing the model file to not be found — with no error, just a silent failure or panic downstream.

**Why it happens:** Rust's `Path::to_str()` returns `None` on non-UTF-8 paths (Windows-specific). The existing code swallows this `None`.

**Consequences:**
- On Windows systems with non-ASCII usernames, model loading silently fails.
- The error manifests as "model not found" or a panic at the `CString::new(...).unwrap()` call that follows.
- No error reaches the Flutter UI — the app just hangs or crashes.

**Prevention:**
- In the Flutter GUI, always use `path_provider`'s temp directory for output (already planned — this avoids user-profile paths for output).
- For model paths (which users specify), add a Dart-side validation that checks whether the path contains only ASCII before passing it to Rust. Show a clear error if not.
- Do not rely on the Rust library to handle this gracefully — it currently does not.

**Warning signs:**
- Windows-only bug reports: "generation never starts."
- User's `%USERPROFILE%` path contains non-Latin characters.

**Phase that must address it:** Phase 2 — add Dart-side path validation before any Rust call.

---

### Pitfall M-5: Yaru on macOS/Windows — Font and Scroll Behavior Differences

**What goes wrong:** Yaru ships the Ubuntu font and uses GTK-style spacing/sizing conventions. On macOS and Windows, Flutter's text rendering pipeline uses the platform's native font fallback chain when the bundled font is not explicitly loaded. If the Yaru setup does not properly load the Ubuntu font asset, text renders in the platform's default font (San Francisco on macOS, Segoe UI on Windows), which has different metrics. This causes layout overflow, clipped labels, and widgets that look correct on Linux but broken elsewhere.

**Why it happens:** Yaru's Flutter package bundles the Ubuntu font, but it must be declared in the app's `pubspec.yaml` assets section to be available. If the developer installs Yaru and forgets to declare the font asset, the font silently falls back.

**Consequences:**
- Text truncation in dropdowns and form labels on macOS/Windows.
- Pixel-level layout differences between platforms.
- Scroll physics differs: macOS uses momentum scrolling; Yaru's default scroll physics are Linux-tuned.

**Prevention:**
- Follow Yaru's setup instructions exactly: add `uses-material-design: true` and the Yaru font assets to `pubspec.yaml`.
- Test the UI on all three target platforms in Phase 1, not just Linux.
- For scroll physics, use `ScrollConfiguration` with platform-adaptive physics explicitly.

**Warning signs:**
- Text appears in a different font on macOS/Windows than on Linux.
- Form labels are clipped on macOS but fine on Linux.
- Dropdown widgets have different heights per platform.

**Phase that must address it:** Phase 1 (mock UI) — catch this during initial cross-platform smoke testing.

---

## Minor Pitfalls

---

### Pitfall m-1: Temporary File Accumulation After Crash

**What goes wrong:** Flutter's `path_provider` `getTemporaryDirectory()` returns a system temp directory, but the OS does NOT clean it on app exit. The planned cleanup (delete temp dir on app close) only runs if the app exits cleanly. If the app crashes (Rust panic, OOM, forced kill), the cleanup code never runs. Each crashed session leaves behind potentially large image files (512×512 to 1024×1024 PNG files from a batch generation can be 50–500MB total).

**Why it happens:** Cleanup is typically implemented in `dispose()` or `AppLifecycleState.detached` handlers, neither of which fires on unclean exit.

**Prevention:**
- On startup, scan the temp directory for any leftover files from previous sessions older than N hours and delete them.
- Name the session temp subdirectory with the process PID: `diffusion_rs_gui_<pid>/`. On startup, check for dirs whose PID no longer exists (using `Process.canRun` or platform-specific check) and delete them.
- Keep generated images small enough to not be catastrophic if leaked (use temp PNG, not uncompressed raw).

**Warning signs:**
- User reports growing disk usage over time.
- Temp directory contains many `diffusion_rs_gui_*` folders.

**Phase that must address it:** Phase 1 — design the session temp dir pattern; Phase 2 — implement the startup cleanup sweep.

---

### Pitfall m-2: Lazy Model Loading Race — Second Generation Before First Finishes

**What goes wrong:** diffusion-rs implements lazy model loading (recent commit `deb776a`) using unsafe pointer storage (`Option<(NonNull<sd_ctx_t>, ...)>`). The comment in CONCERNS.md notes "concurrent access patterns are untested." If the user somehow triggers a second generation (e.g., by clicking Start rapidly, or if the Stop button has a race), the second call may find the model mid-initialization with a partially initialized raw pointer.

**Why it happens:** The lazy init is designed for single-threaded CLI use. In a GUI context with async callbacks and user input events firing concurrently, a TOCTOU (time-of-check-time-of-use) race becomes possible.

**Prevention:**
- In the Flutter UI, disable the Start button immediately upon click and do not re-enable it until the result (success or error) arrives. Never re-enable it on Stop without waiting for the Rust thread to actually finish.
- In Rust, add a `Mutex<bool>` generation_in_progress guard at the FFI boundary that returns an error if a generation is already running.

**Warning signs:**
- Start button can be clicked while generation is running (UI bug).
- Rust segfault with a backtrace touching `sd_ctx_t` initialization.

**Phase that must address it:** Phase 2 — UI disable/enable logic is the first safeguard.

---

### Pitfall m-3: macOS Code Signing — Native Library Not Signed

**What goes wrong:** On macOS, if the Flutter desktop app is distributed outside the App Store (as a `.app` bundle), all bundled dylibs must be ad-hoc signed at minimum. The diffusion-rs `.dylib` compiled by `cargo build` is unsigned. When users download and run the app, Gatekeeper may block loading the library with "dylib cannot be opened because the developer cannot be verified."

**Why it happens:** Cargo does not sign dylibs. The Flutter build process signs the `.app` bundle but may not recursively sign nested dylibs from Cargo unless explicitly configured.

**Prevention:**
- In the release build script, run `codesign --force --sign - <path-to-dylib>` (ad-hoc) or with a Developer ID if distributing to other machines.
- Flutter's macOS build can be configured with a `Podfile`-level hook to sign all native libraries.
- Test on a second Mac that has never run the app in development mode (Gatekeeper is not suppressed there).

**Warning signs:**
- App works on developer's Mac but crashes on another Mac on first launch.
- macOS Console shows `AMFI: <dylib> is not signed` or `dyld: Library not loaded`.

**Phase that must address it:** Phase 2 — add signing to the macOS release build.

---

### Pitfall m-4: Git Submodule Not Initialized — Confusing Build Failures

**What goes wrong:** `sys/stable-diffusion.cpp/` is a git submodule. A developer who clones the monorepo without `--recursive` and then tries to build the GUI Rust crate gets: `No such file or directory: sys/stable-diffusion.cpp/CMakeLists.txt`. This error appears in the `build.rs` CMake output, not as a clear message about the submodule.

**Why it happens:** The submodule requirement is documented in the README but not enforced at the shell/CI level. The error message from CMake is not self-explanatory.

**Prevention:**
- Add a `build.rs` check at the very start: verify `sys/stable-diffusion.cpp/CMakeLists.txt` exists and emit a clear `println!("cargo:warning=Run git submodule update --init --recursive")` if it does not, then `panic!`.
- Document the setup in the GUI's own README (not just the root README).

**Warning signs:**
- Build fails with: `could not find CMakeLists.txt` or similar CMake error.
- The `sys/stable-diffusion.cpp/` directory is empty.

**Phase that must address it:** Phase 2 setup — document in the GUI README.

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| Phase 1 — Mock UI setup | Yaru font not bundled correctly on macOS/Windows | Test on non-Linux platform in Phase 1, not Phase 2 |
| Phase 1 — Stop button design | No cancellation API in C++ backend; UI must communicate "pending stop" not "immediate stop" | Design the UX honestly — "Stop after current step" |
| Phase 1 — Progress bar simulation | Using `Timer.periodic` vs `Stream` — wrong pattern learned in mock becomes wrong pattern in real wiring | Use `Stream`-based mock to mirror the real `StreamSink` API |
| Phase 2 — FFI bridge setup | Cargo workspace + Flutter subfolder: artifact path not found by flutter build | Configure `cargokit` paths explicitly; verify with release build smoke test |
| Phase 2 — First generation call | Sync Rust function blocks main isolate | Verify UI remains responsive by running a separate animation during generation |
| Phase 2 — Progress callbacks | C++ thread calling through FFI into Dart event loop | Use flutter_rust_bridge StreamSink; never store raw Dart port in C++ |
| Phase 2 — Panic safety | 40+ `.unwrap()` calls in diffusion-rs; any unusual input panics | Add `catch_unwind` wrappers; compile with `panic = "abort"` |
| Phase 2 — Windows testing | Non-UTF-8 user profile paths silently dropped | Add Dart-side ASCII path validation before passing to Rust |
| Phase 2 — macOS distribution | Unsigned dylib blocked by Gatekeeper | Ad-hoc sign in release build script |
| Phase 2 — Temp file cleanup | Crash leaves large image files in temp dir | Startup sweep to delete stale session directories |
| Phase 2 — token.txt build dep | Missing token.txt breaks GUI Rust build on fresh checkout | Create empty placeholder; document in GUI README |

---

## Sources

| Source | Provider | Confidence |
|--------|----------|------------|
| flutter_rust_bridge v2 documentation (architecture, isolate model, StreamSink, codegen) | Training knowledge (cutoff Aug 2025) | MEDIUM |
| diffusion-rs codebase analysis (CONCERNS.md: unsafe FFI, unwrap, token.txt, lazy loading) | Direct codebase read | HIGH |
| path_provider Flutter package behavior on desktop platforms | Training knowledge | MEDIUM |
| Yaru Flutter package cross-platform behavior | Training knowledge | MEDIUM |
| macOS Gatekeeper / dylib signing requirements | Training knowledge | MEDIUM |
| Rust panic-across-FFI undefined behavior (Rustonomicon) | Training knowledge | HIGH |
| Cargo workspace + cdylib linking patterns | Training knowledge | MEDIUM |
