---
title: Concerns
last_mapped: 2026-06-18
---

# Concerns

## High Priority

### Pervasive `.unwrap()` Usage
- **Location:** `src/api.rs`, `src/preset.rs`, `src/modifier.rs`, `cli/src/main.rs`
- **Count:** 40+ `.unwrap()` / `.expect()` calls across the library
- **Risk:** Any unexpected state (invalid UTF-8 in path, poisoned lock, builder field not set) causes a panic instead of returning an error to the caller
- **Examples:**
  - `src/api.rs:1303` — `PathBuf::from(value.0.to_str().unwrap())` (panics on non-UTF-8 paths)
  - `src/util.rs:13` — `guard.write().unwrap()` (panics if RwLock is poisoned)
  - `src/api.rs:619` — `valid_loras.get(&spec.file_name).unwrap()` (panics if LoRA not found, though preceded by a check)
- **Mitigation:** Library should return `Result` consistently; only CLI binaries should call `.unwrap()`

### Unsafe FFI Boundary
- **Location:** `src/api.rs` — multiple `unsafe` blocks and functions
- **Risk:** Raw pointer arithmetic and `slice::from_raw_parts` from C++ allocated memory; no bounds checking; use-after-free possible if C++ context is freed while Rust holds a pointer
- **Key areas:**
  - `src/api.rs:788` — `unsafe fn upscaler_ctx()` returns raw `*mut upscaler_ctx_t`
  - `src/api.rs:1318` — `unsafe fn upscale()` standalone function
  - `src/api.rs:1663` — `slice::from_raw_parts(img.data, len)` from C++ image struct
  - `unsafe extern "C"` callbacks for progress and preview at `src/api.rs:1513`, `src/api.rs:1540`

### `token.txt` Compile-Time Dependency
- **Location:** `src/preset.rs` and `src/modifier.rs` tests
- **Risk:** `include_str!("../token.txt")` is evaluated at **compile time**, so the file must exist even when the tests are `#[ignore]`d. This breaks fresh checkouts and CI without the file.
- **Impact:** Currently `token.txt` is committed (visible in root directory listing), which may expose a real HuggingFace token in git history

## Medium Priority

### Almost Zero Automated Test Coverage
- **Details:** ~61 of ~64 tests are `#[ignore]`d, requiring real model weights. Only ~3 tests run in CI.
- **Risk:** Regressions in builder logic, FFI parameter mapping, or error handling go undetected
- **Mitigation:** Add non-ignored unit tests for builder defaults, validation, and error paths

### Lazy Model Loading Complexity
- **Location:** `src/api.rs` — `DiffusionModel` loads C++ context on first use, caches raw pointers in `Option<(NonNull<sd_ctx_t>, ...)>`
- **Risk:** The lazy init pattern combined with `unsafe` pointer storage is fragile; concurrent access patterns are untested
- **Note:** Based on `deb776a feat: lazily load params from disk` recent commit, this is actively changing

### Backend-Specific VRAM Budgets Added Recently
- **Location:** `src/api.rs` model config, based on `ae1cc2f feat: support backend-specific max-vram budgets`
- **Risk:** New feature without automated tests; behavior varies across GPU backends

### Windows Path Handling
- **Location:** `src/api.rs:1291`, `SafePathBuf` conversions
- **Risk:** `unwrap_or_default()` silently drops non-UTF-8 paths on Windows (where paths can contain non-UTF-8); errors are swallowed
- **Pattern:** `CString::new(value.to_str().unwrap_or_default()).unwrap()`

## Low Priority / Observations

### Git Submodule for C++ Library
- `sys/stable-diffusion.cpp/` is a git submodule — developers must `git clone --recursive` or run `git submodule update --init`
- Not documented prominently in README; can cause confusing build failures

### SYCL Backend Commented Out in CI
- `.github/workflows/test.yml` has the entire `build-sycl` job commented out
- OneAPI dependency is complex; unknown if SYCL feature actually works currently

### `num_cpus` Dependency
- `num_cpus` crate is deprecated in favor of `std::thread::available_parallelism` (stabilized in Rust 1.59)
- Low risk, but adds a dependency for functionality now in std

### No `#![deny(clippy::unwrap_used)]` Lint
- No lint configuration enforcing `Result`-based error handling
- Adding `#![warn(clippy::unwrap_in_result)]` would catch the most dangerous cases

### `dist-workspace.toml` / cargo-dist Setup
- Release tooling (`cargo-dist`, `release-plz`) is configured; appears healthy based on recent release commits
- SYCL is excluded from distribution builds (consistent with commented-out CI)
