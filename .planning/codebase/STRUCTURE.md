---
title: Directory Structure
last_mapped: 2026-06-18
---

# Directory Structure

## Root Layout

```
diffusion-rs/
├── src/                    # Main library crate (diffusion-rs)
│   ├── lib.rs              # Public re-exports
│   ├── api.rs              # Core FFI bridge + generation logic
│   ├── preset.rs           # Preset enum + PresetConfig builder
│   ├── preset_builder.rs   # High-level PresetBuilder (user-facing API)
│   ├── modifier.rs         # Modifier trait + implementations (LoRA, ControlNet, etc.)
│   └── util.rs             # HuggingFace token helpers
├── sys/                    # FFI bindings crate (diffusion-rs-sys)
│   ├── src/
│   │   └── lib.rs          # bindgen-generated bindings
│   ├── build.rs            # cmake build script for stable-diffusion.cpp
│   ├── wrapper.h           # C header wrapper for bindgen
│   └── stable-diffusion.cpp/  # Git submodule (upstream C++ library)
├── cli/                    # Binary crate (diffusion-rs-cli)
│   └── src/
│       └── main.rs         # CLI entry point (clap-based)
├── .github/
│   └── workflows/
│       ├── test.yml        # CI: fmt + multi-platform builds + tests
│       ├── release.yml     # CD: cargo-dist release pipeline
│       └── release-plz.yml # Automated release-plz on main
├── Cargo.toml              # Workspace root + diffusion-rs package
├── Cargo.lock
├── cliff.toml              # git-cliff changelog config
├── dist-workspace.toml     # cargo-dist distribution config
├── release-plz.toml        # release-plz config
├── token.txt               # HuggingFace token (gitignored, referenced in tests)
└── CHANGELOG.md
```

## Crate Responsibilities

| Crate | Name | Role |
|-------|------|------|
| Root | `diffusion-rs` | High-level Rust API, published to crates.io |
| `sys/` | `diffusion-rs-sys` | Raw FFI bindings, published to crates.io |
| `cli/` | `diffusion-rs-cli` | Binary CLI, distributed via cargo-dist, not published |

## Key Files

### `src/lib.rs`
Public re-export surface. Re-exports everything from `api`, `preset`, `preset_builder`, `modifier`, and `util`.

### `src/api.rs`
The most complex file. Contains:
- `DiffusionModel` — holds raw C++ context pointers, lazy-loads on first use
- `ModelConfig` — builder for model initialization parameters (backend, VRAM, threads, etc.)
- `GenerationParams` — builder for per-inference parameters (prompt, steps, guidance, size, etc.)
- `gen_img()` — top-level generation function
- FFI helper types: `SafeCString`, `SafePathBuf`, raw pointer management
- `unsafe` callbacks for progress and preview

### `src/preset.rs`
- `Preset` enum — one variant per supported model family (SD1.x, SD2.x, SDXL, SD3, Flux, etc.)
- `PresetConfig` — resolves a `Preset` to concrete file paths and model parameters
- Weight quantization enums per model family (`Flux1Weight`, `Flux1MiniWeight`, etc.)
- Contains tests for preset resolution (all `#[ignore]`d, require real models)

### `src/preset_builder.rs`
- `PresetBuilder` — high-level builder consumed by end users
- Calls into `preset.rs` resolution, then dispatches to `api.rs`

### `src/modifier.rs`
- `Modifier` trait — applied to `GenerationParams` before inference
- Implementations: `LoRA`, `ControlNet`, `UpscaleModifier`, `IPAdapter`, etc.
- Contains unit tests for modifier-preset combinations (all `#[ignore]`d)

### `sys/build.rs`
Drives `cmake` to compile `stable-diffusion.cpp` and links the resulting static library. Handles feature flags (`cuda`, `metal`, `vulkan`, `hipblas`, `sycl`), cross-platform paths, and Windows-specific quirks.

## Where to Add New Code

| What | Where |
|------|-------|
| New model preset | `src/preset.rs` — add variant to `Preset`, add resolution arm to `PresetConfig` |
| New weight quantization | `src/preset.rs` — add variant to the model's weight enum |
| New modifier (LoRA variant, adapter, etc.) | `src/modifier.rs` — implement `Modifier` trait |
| New CLI flag | `cli/src/main.rs` — extend clap struct |
| New generation parameter | `src/api.rs` — extend `GenerationParams` builder |
| New GPU backend | `Cargo.toml` features + `sys/build.rs` + `src/api.rs` backend enum |

## Naming Conventions

- Types: `PascalCase` (e.g., `PresetBuilder`, `ModelConfig`, `SafeCString`)
- Functions/methods: `snake_case` (e.g., `gen_img`, `set_hf_token`)
- Features: lowercase single word or compound (e.g., `cuda`, `hipblas`, `metal`)
- Test functions: `test_<model_name>` pattern (e.g., `test_flux_1_schnell`)
- Enums variants: `PascalCase` with version numbers as suffix (e.g., `StableDiffusion1_4`, `Flux1Dev`)

## Special Directories

- `sys/stable-diffusion.cpp/` — git submodule, do not edit directly
- `.github/workflows/` — CI/CD pipelines, one per concern (test, release, release-plz)
- `token.txt` — gitignored but referenced by `#[ignore]` tests via `include_str!("../token.txt")`
