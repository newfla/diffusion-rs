# External Integrations

**Analysis Date:** 2026-06-18

## APIs & External Services

**HuggingFace Hub:**
- HuggingFace Model Hub - Primary source for downloading pre-trained Stable Diffusion models
  - SDK/Client: `hf_hub` (v0.4.2 with `ureq` for HTTP)
  - Auth: `HF_TOKEN` environment variable (optional for public models, required for gated models)
  - Implementation: `src/util.rs` - `set_hf_token()` and `download_file_hf_hub()`
  - Features: Model discovery, direct download to local cache with automatic extraction

**Model Repository:**
- Supported Model Sources:
  - Stable Diffusion models (v1.4, v1.5, v2.1, v3.5 variants)
  - SDXL (SDXLTurbo, SDXS variants)
  - Flux (Flux 1.0 and Flux 2.0 variants)
  - Specialized models (ANIVERSE, Chroma, DiffInstruct-Star, Qwen Image, etc.)
  - All models stored as GGUF/safetensors/PyTorch weights in HuggingFace repos

## Data Storage

**Databases:**
- None - Stateless image generation library

**File Storage:**
- HuggingFace Hub Cache - Remote model repository
  - Client: `hf_hub::api::sync::ApiBuilder`
  - Cache Location: User's HuggingFace cache directory (auto-managed by hf-hub crate)
  - Model Files: Downloaded on-demand, cached locally for subsequent runs

**Local File System:**
- Model weights cached in `~/.cache/huggingface/` (default)
- Generated images saved to user-specified output directory
- EXIF metadata embedded in output PNG/JPEG images via `little_exif`

**Caching:**
- Application-level: None (stateless per invocation)
- Build-level: CMake build artifacts in `target/` directory
- Model-level: HuggingFace hub caching handled by `hf_hub` crate

## Authentication & Identity

**Auth Provider:**
- HuggingFace API Token (optional, environment variable)
  - Implementation: `src/util.rs`
  - Mechanism: Bearer token in HTTP Authorization header
  - Required for: Private/gated models on HuggingFace
  - Optional for: Public model access

**Token Management:**
- Static global token via `set_hf_token(&str)` - sets token once per runtime
- Stored in thread-safe `OnceLock<RwLock<String>>`
- Environment variable fallback: `hf_hub` automatically checks `HF_TOKEN` env var

## Monitoring & Observability

**Error Tracking:**
- None configured - errors returned as `Result` types via `thiserror` custom error enum

**Logs:**
- Console output (stderr) - Progress callbacks via C++ backend
- Benchmark timing: `execution-time` crate for performance measurement
- EXIF metadata: Timestamps embedded in generated images via `chrono::Local`

**Progress Callbacks:**
- C++ callback: `sd_set_progress_callback()` - inference progress tracking
- C++ callback: `sd_set_preview_callback()` - intermediate image preview generation
- Rust API: Message channel (`std::sync::mpsc::Sender`) for sending progress updates to caller

## CI/CD & Deployment

**Hosting:**
- Crates.io - Official Rust package registry for library distribution
- Docs.rs - Automated documentation hosting
- GitHub Releases - Binary distribution (configured in `.github/workflows/release.yml`)

**CI Pipeline:**
- GitHub Actions for multi-platform testing
- Jobs:
  - `cargo-fmt` - Code style checking (required)
  - `build-no-features` - CPU-only baseline (Ubuntu, macOS, Windows)
  - `build-vulkan` - Vulkan backend validation (Ubuntu, macOS)
  - `build-metal` - Metal GPU validation (macOS only)
  - `build-cuda` - NVIDIA CUDA validation (Ubuntu 22.04 with CUDA 12.8)
  - `build-rocm` - AMD HIP validation (Ubuntu with ROCm 6.1.2)
  - Skipped: SYCL tests (commented out due to build time constraints)

**Release Process:**
- Release-plz automation for versioning and changelog
- Git tags: `v{{ version }}` format
- GitHub releases disabled (publish to crates.io only)
- Changelog: Generated from commit messages via git-cliff

## Environment Configuration

**Required env vars (Optional - Runtime):**
- `HF_TOKEN` - HuggingFace Hub authentication token (for gated models)

**Required env vars (Build-Time for GPU features):**
- `CUDA_PATH` - CUDA toolkit location (Windows CUDA builds)
- `CUDA_COMPUTE_CAP` - CUDA compute capability target (default inferred from GPU)
- `VULKAN_SDK` - Vulkan SDK path (required on Windows, optional on Unix)
- `HIP_PATH` - AMD ROCm installation path (defaults to `/opt/rocm` on Unix)
- `GFX_NAME` - AMD GPU architecture (e.g., "gfx1100")
- `ONEAPI_ROOT` - Intel oneAPI toolkit root (SYCL feature)

**Secrets location:**
- HuggingFace token: `HF_TOKEN` environment variable (typically loaded from `.env` or CI secrets)
- No hardcoded credentials in codebase

## Webhooks & Callbacks

**Incoming:**
- None - Library API only (no server component)

**Outgoing:**
- C++ Backend Callbacks:
  - Progress callback (`sd_set_progress_callback`) - Reports inference step progress
  - Preview callback (`sd_set_preview_callback`) - Returns intermediate image data mid-inference
  - Both routed through Rust via `std::sync::mpsc::Sender<Progress>`

## Model Presets

**Included Preset Models (Auto-Downloaded from HuggingFace):**

Primary Models:
- Flux 1.0 Schnell / Dev (Text-to-image, instruction-following)
- Flux 1.0 Mini (Lightweight variant)
- Flux 2.0 (Text-to-image)
- Flux 2.0 Klein (4B and 9B variants)
- Stable Diffusion v1.4, v1.5, v2.1
- SDXL 1.0 Base / Turbo
- SDXS 512 DreamShaper
- Stable Diffusion 3.5 (Medium, Large, Large Turbo)

Specialized Models:
- Chroma / Chroma Radiance (Style synthesis)
- NitroSD Realism / Vibrant (Photography/artistic styles)
- DiffInstruct-Star (Instruction-following)
- SSD-1B (Speed-focused)
- ZImageTurbo / TwinFlow Z Image Turbo Exp (Fast image generation)
- Qwen Image / Ovis Image (Multimodal image understanding)
- ANIVERSE / Anima / Anima2 (Animation-specific)
- Ernie Image (Text-to-image generation)
- LongCat Image (Extended context)

All presets download models automatically from HuggingFace on first run via `hf-hub` client.

## Model Configuration

**Weight Quantization:**
- F32 - Full precision (32-bit float)
- F16 - Half precision (16-bit float)
- Q8_0 / Q8_1 - 8-bit quantization
- Q5_0 / Q5_1 - 5-bit quantization
- Q4_0 / Q4_1 - 4-bit quantization
- Q3_K / Q3_K_S / Q3_K_M / Q3_K_L - 3-bit K-quant variants
- Q2_K / Q6_K - Alternative quantization schemes

Each preset supports specific weight types for memory/speed optimization.

**Inference Parameters:**
- Scheduler: Euler, DPM++, LCM, Heun, DPM (configurable per run)
- Sampling Method: Ancestral, Automatic, Default (model-specific)
- Guidance: CFG scale (1.0-20.0), classifier-free guidance strength
- Steps: Inference step count (10-100+)
- Seed: Reproducible generation via RNG control

---

*Integration audit: 2026-06-18*
