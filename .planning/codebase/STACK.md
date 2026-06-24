# Technology Stack

**Analysis Date:** 2026-06-18

## Languages

**Primary:**
- Rust (Edition 2024) - Core library and FFI bindings for Stable Diffusion

**Secondary:**
- C/C++ - Stable Diffusion.cpp backend (submodule: `stable-diffusion.cpp/`)

## Runtime

**Environment:**
- Rust Toolchain (Stable) - Compiled to native binaries for Windows, macOS, Linux

**Package Manager:**
- Cargo - Rust package manager
- Lockfile: Present (`Cargo.lock`)

## Frameworks

**Core:**
- `diffusion-rs-sys` (v0.1.20) - FFI bindings to stable-diffusion.cpp using bindgen
- `stable-diffusion.cpp` - Submodule providing the core C++ inference engine

**Build/Dev:**
- `cmake` (v0.1.51) - CMake integration for building C++ backend during cargo build
- `bindgen` (v0.71.1) - Auto-generates Rust FFI bindings from C/C++ headers
- `clap` (v4.5.53) - CLI argument parsing with derive macros
- `derive_builder` (v0.20.2) - Builder pattern macro generation

## Key Dependencies

**Critical:**
- `hf_hub` (v0.4.2 with `ureq`) - HuggingFace Hub API client for downloading models
- `image` (v0.25.5) - Image encoding/decoding and manipulation (PNG, JPEG, WebP, AVIF)
- `libc` (v0.2.161) - C standard library bindings

**Infrastructure:**
- `chrono` (v0.4.42) - Date/time handling (EXIF metadata generation)
- `thiserror` (v2.0.3) - Error handling and custom error types
- `strum` (v0.27) - Enum string conversion macros (`EnumString`, `VariantNames`)
- `subenum` (v1.1.3) - Conditional enum variant grouping
- `little_exif` (v0.6.21) - EXIF metadata writing to generated images
- `walkdir` (v2.5.0) - Directory traversal for batch operations
- `num_cpus` (v1.16.0) - CPU count detection for thread pool sizing
- `execution-time` (v0.3.1) - Performance timing instrumentation

**Build Dependencies:**
- `fs_extra` (v1.3.0) - File system operations during build process

## Features (Compile-Time)

**GPU Acceleration Backends:**
- `cuda` - NVIDIA CUDA support (CUDA 12.8+, compute capability configurable via `CUDA_COMPUTE_CAP`)
- `metal` - Apple Metal GPU (auto-enabled on macOS unless Vulkan is selected)
- `vulkan` - Cross-platform Vulkan (Windows, macOS, Linux)
- `hipblas` - AMD ROCm HIP support (requires `HIP_PATH` env var)
- `sycl` - Intel oneAPI DPC++ support (Intel SYCL backend, requires `ONEAPI_ROOT`)

**Default Behavior:**
- CPU-only mode when no GPU features enabled
- Metal automatically enabled on macOS unless Vulkan feature is specified
- CMake builds with `GGML_OPENMP=OFF`, `SD_BUILD_SHARED_LIBS=OFF`

## Configuration

**Build Configuration Files:**
- `Cargo.toml` - Workspace configuration with shared dependencies
- `sys/Cargo.toml` - FFI bindings package
- `cli/Cargo.toml` - CLI tool package
- `sys/build.rs` - Custom C++ build orchestration
- `sys/wrapper.h` - Bindgen input header
- `dist-workspace.toml` - Distribution/release configuration
- `cliff.toml` - Git changelog generation rules

**Environment Variables (Build-Time):**
- `CUDA_PATH` - CUDA toolkit installation path (Windows)
- `CUDA_COMPUTE_CAP` - Target CUDA compute capability (e.g., "75" for 7.5)
- `VULKAN_SDK` - Vulkan SDK installation path (Windows requires this explicitly)
- `HIP_PATH` - AMD ROCm installation path (defaults to `/opt/rocm` on Unix)
- `GFX_NAME` - AMD GPU architecture target (e.g., "gfx1100")
- `ONEAPI_ROOT` - Intel oneAPI root path (SYCL feature)
- `DOCS_RS` - Skips C++ compilation on docs.rs builds

## Platform Requirements

**Development:**
- Rust 1.56+ (Edition 2024 requires nightly or upcoming stable)
- CMake 3.15+
- Clang/LLVM (for bindgen)
- C++ compiler with C++14 support minimum
- macOS: Xcode Command Line Tools (Metal/Accelerate frameworks)
- Windows: MSVC compiler, Ninja build system
- Linux: GCC/Clang, libc-dev

**Production:**
- Deployment: Windows (x86-64), macOS (arm64/x86-64), Linux (x86-64)
- GPU libraries installed for selected backend (CUDA, Metal, Vulkan, HIP, or SYCL)
- Minimum 8GB RAM recommended for most models
- GPU with 4GB+ VRAM for acceleration (GPU feature dependent)

**Release & CI/CD:**
- GitHub Actions for testing on Ubuntu 22.04, macOS latest, Windows latest
- Release-plz for automated versioning and changelog
- Crates.io publishing (registry)
- Documentation hosted on docs.rs

---

*Stack analysis: 2026-06-18*
