<!-- refreshed: 2026-06-18 -->
# Architecture

**Analysis Date:** 2026-06-18

## System Overview

```text
┌─────────────────────────────────────────────────────────────┐
│                       CLI Application                        │
│                   `cli/src/main.rs`                          │
│                   (Command-line interface)                   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  High-Level Rust API                         │
│  ConfigBuilder, ModelConfigBuilder, Preset System            │
│  `src/api.rs` (1800+ lines), `src/preset.rs`, `src/preset_builder.rs`
└────────┬──────────────────────────┬──────────────────────────┘
         │                          │
         ▼                          ▼
┌─────────────────────┐  ┌──────────────────────────┐
│  Configuration      │  │  Model Presets &         │
│  Management         │  │  Modifiers               │
│  `src/api.rs`       │  │  `src/preset.rs`         │
│  - Config           │  │  `src/modifier.rs`       │
│  - ModelConfig      │  │  `src/preset_builder.rs` │
│  - Cache params     │  │  `src/util.rs`           │
└────────┬────────────┘  └──────────┬───────────────┘
         │                          │
         └──────────────┬───────────┘
                        ▼
         ┌─────────────────────────────────┐
         │  FFI Layer to stable-diffusion  │
         │  .cpp (C++ backend)             │
         │                                 │
         │  `diffusion-rs-sys` crate       │
         │  `sys/src/lib.rs` (bindgen)     │
         └─────────────────────────────────┘
                        │
                        ▼
         ┌─────────────────────────────────┐
         │  stable-diffusion.cpp           │
         │  (Submodule C++ implementation)  │
         │  - Model inference               │
         │  - Sampling methods              │
         │  - Upscaling                     │
         │  - Cache strategies              │
         └─────────────────────────────────┘
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| CLI Parser | Parse command-line arguments, preset selection, parameter overrides | `cli/src/main.rs` |
| Config Builder | Build and validate generation configs (prompt, dimensions, sampling) | `src/api.rs` (ConfigBuilder) |
| ModelConfig Builder | Build and validate model configs (paths, backends, caching, LoRAs) | `src/api.rs` (ModelConfigBuilder) |
| Preset System | Pre-configured model setups with auto-downloading from Hugging Face | `src/preset.rs` |
| Preset Builder | Chain presets with modifiers (LoRAs, upscalers, VAE fixes) | `src/preset.rs` (PresetBuilder) |
| Modifiers | Composable functions to enhance presets (faster decoding, VRAM reduction) | `src/modifier.rs` |
| API Entry Points | Core generation functions (gen_img, gen_img_with_progress) | `src/api.rs` |
| Utility Helpers | HuggingFace Hub integration, file downloads | `src/util.rs` |
| FFI Bindings | Type-safe wrapper around stable-diffusion.cpp C API | `diffusion-rs-sys/sys/src/lib.rs` |

## Pattern Overview

**Overall:** Builder-first, type-safe wrapper pattern with modular configuration composition.

**Key Characteristics:**
- **Builder pattern** for all configs — ConfigBuilder and ModelConfigBuilder use `derive_builder` crate
- **Preset system** — Pre-baked model configurations auto-download from Hugging Face
- **Composable modifiers** — Apply LoRAs, VAE fixes, caching strategies via chaining
- **Safe FFI wrapper** — `CLibString` and `CLibPath` types manage C string lifetimes
- **Extensible backend support** — Compile-time feature flags for CUDA, Metal, Vulkan, HIP, SYCL

## Layers

**CLI Layer:**
- Purpose: User interaction, argument parsing, progress display
- Location: `cli/src/main.rs`
- Contains: Command-line argument parsing (clap), file I/O, progress reporting
- Depends on: diffusion-rs API, util for HF token
- Used by: End users via binary

**Configuration & Builder Layer:**
- Purpose: Type-safe configuration construction and validation
- Location: `src/api.rs` (ConfigBuilder ~1200 lines, ModelConfigBuilder ~800 lines)
- Contains: Builder structs, validation logic, C interop wrappers (CLibString, CLibPath)
- Depends on: Derives from `derive_builder`, thiserror for error handling
- Used by: PresetBuilder, CLI, user code

**Preset & Modifier Layer:**
- Purpose: Pre-configured model setups and composable enhancements
- Location: `src/preset.rs`, `src/preset_builder.rs`, `src/modifier.rs`
- Contains: Enum-based preset definitions, preset factory functions, modifier functions
- Depends on: ConfigBuilder, ModelConfigBuilder, util for downloads, hf_hub
- Used by: CLI, user code via PresetBuilder

**API Layer:**
- Purpose: Core image generation entry points
- Location: `src/api.rs` (functions: gen_img, gen_img_with_progress, gen_img_maybe_progress)
- Contains: Main generation logic (unsafe), image I/O, upscaling, caching setup
- Depends on: diffusion-rs-sys, image crate for PNG operations, little_exif for metadata
- Used by: CLI, user code

**FFI Layer:**
- Purpose: Safe Rust bindings to stable-diffusion.cpp C API
- Location: `diffusion-rs-sys/sys/src/lib.rs` (generated by bindgen)
- Contains: Auto-generated FFI bindings from wrapper.h
- Depends on: C++ backend via CMake build script
- Used by: api.rs

**Utility Layer:**
- Purpose: Helper functions for external integrations
- Location: `src/util.rs`
- Contains: HuggingFace Hub API wrapper, token management
- Depends on: hf_hub crate
- Used by: preset_builder.rs, modifier.rs, CLI

## Data Flow

### Primary Request Path (Text-to-Image Generation)

1. **CLI Entry** (`cli/src/main.rs:127`) — Args::parse() captures preset, prompt, parameters
2. **Preset Construction** (`cli/src/main.rs:135`) — get_preset() calls Preset::try_configs_builder()
3. **Builder Configuration** (`src/preset_builder.rs:23+`) — Model downloaded via hf_hub, configs created
4. **Modifier Application** (`src/preset.rs:451`) — TryFrom<PresetConfig> applies modifier chain
5. **Config Building** (`src/preset.rs:440`) — ConfigBuilder::build() and ModelConfigBuilder::build() create immutable configs
6. **Model Context Creation** (`src/api.rs:809`) — ModelConfig::diffusion_ctx() lazily initializes C context via new_sd_ctx
7. **Image Generation** (`src/api.rs:1360`) — gen_img_maybe_progress() calls generate_image(sd_ctx, &params) via FFI
8. **Sampling Loop** (stable-diffusion.cpp) — C++ backend performs diffusion steps, optional progress callbacks
9. **Upscaling** (`src/api.rs:1318`) — Optional ESRGAN upscaling if configured
10. **Image Save** (`src/api.rs:1660`) — PNG output with EXIF metadata (generation parameters)

### Image-to-Image + Inpainting Flow

1. Steps 1-5 identical (configuration)
2. **Init Image Load** (`src/api.rs:1449`) — image::open() reads init image to RGB8
3. **Mask Handling** (`src/api.rs:1460`) — Mask loaded as Luma8 or synthesized as white mask if missing
4. **Reference Images** (`src/api.rs:1490`) — Optional reference images for in-context conditioning (Flux2, etc.)
5. **Generation with Constraints** (`src/api.rs:1632`) — generate_image() called with init_image + mask_image + ref_images
6. Rest as primary flow (upscaling, save)

### Caching Strategy Flow (Optional)

1. **Cache Selection** (`cli/src/main.rs:100`) — User selects cache mode (UCACHE, DBCACHE, SPECTRUM, etc.)
2. **Cache Config** (`src/api.rs:1155+`) — ConfigBuilder methods set cache params (thresholds, warmup steps)
3. **Cache Struct Population** (`src/api.rs:1568`) — config.cache.0 passed to generate_image() as sd_cache_params_t
4. **Backend Cache Execution** (stable-diffusion.cpp) — C++ applies caching during inference

**State Management:**
- **Immutable after build** — Config and ModelConfig are built once, then passed as references to gen_img
- **Lazy context initialization** — Diffusion context created on first gen_img call (cached in ModelConfig)
- **Context reuse** — Same ModelConfig instance reused for multiple generations (supports img2img sequences)
- **Context cleanup** — Drop impl on ModelConfig frees sd_ctx and upscaler_ctx (RAII pattern)

## Key Abstractions

**Builder Pattern:**
- Purpose: Enforce required fields (model or diffusion_model), provide sensible defaults, validate interdependencies
- Examples: `ConfigBuilder`, `ModelConfigBuilder`, `HiresParamsBuilder`, cache param builders
- Pattern: derive_builder with custom validation in build_fn(validate = "Self::validate")

**Preset Enum:**
- Purpose: Type-safe enumeration of pre-configured models (30+ variants)
- Examples: Preset::StableDiffusion1_4, Preset::Flux1Dev(Flux1Weight), Preset::Chroma(ChromaWeight)
- Pattern: Match arms in try_configs_builder() dispatch to preset_builder functions

**Modifier Functions:**
- Purpose: Compose enhancements as FnOnce closures chained via PresetBuilder::with_modifier()
- Examples: real_esrgan_x4plus_anime_6_b(), sdxl_vae_fp16_fix(), taesd(), lcm_lora_sd_1_5()
- Pattern: FnOnce(ConfigsBuilder) -> Result<ConfigsBuilder, ApiError>

**Weight Type Subenum:**
- Purpose: Model-specific quantization options (F32, F16, Q4_0, Q8_0, etc.)
- Examples: Flux1Weight::Q2_K, NitroSDRealismWeight::Q8_0
- Pattern: subenum crate generates type-safe subsets per model (Flux1MiniWeight has only compatible types)

**BackendDevice & Module Enums:**
- Purpose: Target-specific GPU allocation and device selection
- Examples: BackendDevice::CUDA0, BackendDevice::VULKAN0, Module::Unet, Module::Vae
- Pattern: HashMap<Module, BackendDevice> maps compute/param modules to specific backends

**Cache Param Structs:**
- Purpose: Algorithm-specific caching configurations
- Examples: SpectrumCacheParams, UCacheParams, DbCacheParams
- Pattern: builder_pattern on each, converted to C struct before passing to generate_image()

## Entry Points

**CLI Binary:**
- Location: `cli/src/main.rs:127` (main function)
- Triggers: User runs `diffusion-rs-cli <preset> "<prompt>" [options]`
- Responsibilities: Parse args, set HF token, construct preset, apply modifiers, call gen_img with progress

**Library API:**
- **gen_img** (`src/api.rs:1356`) — Synchronous image generation, no progress
- **gen_img_with_progress** (`src/api.rs:1347`) — Generation with progress channel (mpsc::Sender)
- Both accept immutable Config and mutable ModelConfig references

**PresetBuilder:**
- Location: `src/preset.rs:422+`
- Triggers: User code calls PresetBuilder::default().preset(Preset::X).prompt("...").build()
- Responsibilities: Auto-download models, chain modifiers, build and validate configs

## Architectural Constraints

- **Mutable ModelConfig required** — gen_img() needs &mut ModelConfig to cache sd_ctx and upscaler_ctx (violation of pure Rust, justified by FFI safety)
- **Unsafe blocks concentrated in api.rs** — All FFI calls (new_sd_ctx, generate_image, upscale) marked unsafe; safety guaranteed by stable-diffusion.cpp
- **Single-threaded inference** — Diffusion model runs on single thread per context; n_threads param controls sample loop parallelism only
- **Global HF token via OnceLock** — `src/util.rs` uses static TOKEN: OnceLock<RwLock<String>> for thread-safe token sharing
- **No circular imports** — Clean dependency hierarchy: util < preset_builder < preset < api < cli (no cycles)
- **CString lifetime management** — CLibString and CLibPath hold ownership; safe to pass as *const c_char to C functions

## Anti-Patterns

### Unused LoRA Storage Struct Fields

**What happens:** `LoraStorage` is Vec<(CLibPath, LoraSpec)> but LoraSpec metadata (is_high_noise, multiplier) is duplicated in sd_lora_t when building the C struct (`src/api.rs:1558`). The multiplier and is_high_noise values are read from LoraSpec during generation, not from the persistent storage.

**Why it's wrong:** Changes to LoraSpec after build won't affect inference; data is copied at generation time. This creates a false sense of mutability.

**Do this instead:** Store LoraSpec immutably as part of ModelConfig, document that modifications require rebuilding, or expose a mutation API that invalidates cached contexts.

### Panics in Hires Configuration

**What happens:** `ModelConfigBuilder::hires_params()` (`src/api.rs:687`) panics if invalid combinations are provided (Upscaler::SD_HIRES_UPSCALER_MODEL without custom_model path).

**Why it's wrong:** Panicking in builder methods violates builder pattern expectations; should return Result.

**Do this instead:** Return Result<&mut Self, ConfigBuilderError> and let caller handle validation errors.

### Magic Constants for Cache Defaults

**What happens:** ConfigBuilder::cache_init() hardcodes all cache parameter defaults (cache.Fn_compute_blocks = 8, warmup = 4, etc.) with no explanation.

**Why it's wrong:** No way to discover or adjust these defaults without reading source code; makes caching hard to understand.

**Do this instead:** Define named constants at module level (e.g., DEFAULT_DBCACHE_FN_BLOCKS = 8) with documentation explaining each.

## Error Handling

**Strategy:** Result-based error handling with thiserror; FFI errors caught via null pointer checks; validation at builder stage.

**Patterns:**
- ConfigBuilder::build() returns Result<Config, ConfigBuilderError> with validation_fn(validate = "Self::validate")
- PresetBuilder::build() converts ApiError from downloads to ConfigBuilderError::ValidationError
- gen_img() returns Result<(), DiffusionError> with enum variants: Forward, StoreImages, Io, Upscaler
- Null pointer check after generate_image() signals OOM/backend failure; returns Err(DiffusionError::Forward)

## Cross-Cutting Concerns

**Logging:** None currently. Uses println! in CLI for status messages only.

**Validation:** 
- ConfigBuilder validates output path/batch_count compatibility (output is file for batch=1, directory for batch>1)
- ModelConfigBuilder validates that model OR diffusion_model is set
- HiresParamsBuilder panics on invalid upscaler mode (should be Result)

**Authentication:**
- HuggingFace token stored in thread-safe static OnceLock<RwLock<String>>
- Token optional; models marked as requiring access will fail download with ApiError if token missing
- CLI accepts --token flag to set before config building

**File I/O:**
- Images saved as RGB8 PNG via image crate
- EXIF metadata (generation parameters) written via little_exif
- Model files cached by hf_hub in ~/.cache/huggingface/hub/ by default

**Memory Management:**
- CLibString and CLibPath use CString::from_str internally with .unwrap() (panics on embedded nulls)
- Image buffers allocated as Vec<u8>, passed to C as *mut u8 pointers
- Reference image storage kept in Vec<Vec<u8>> to outlive FFI call scope

---

*Architecture analysis: 2026-06-18*
