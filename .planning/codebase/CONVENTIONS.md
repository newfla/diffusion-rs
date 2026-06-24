# Coding Conventions

**Analysis Date:** 2026-06-18

## Naming Patterns

**Files:**
- Module files are lowercase with underscores: `preset.rs`, `api.rs`, `modifier.rs`, `util.rs`, `preset_builder.rs`
- Match Rust convention: snake_case for file names

**Functions:**
- Public functions use snake_case: `gen_img()`, `gen_img_with_progress()`, `download_file_hf_hub()`, `set_hf_token()`
- Private/internal methods follow same convention
- Builder methods use snake_case with descriptive names: `embeddings()`, `lora_models()`, `n_threads()`, `hires_params()`, `max_vram()`

**Variables:**
- Local variables use snake_case: `current_image`, `valid_loras`, `backend_map`, `cache_params`, `mask_buffer`
- Tuple destructuring names follow patterns like `(config, model_config)` or `(ConfigBuilder, ModelConfigBuilder)`
- Collections have plural names: `files`, `specs`, `modifiers`, `loras`, `standard`, `high_noise`

**Types:**
- Enums are PascalCase: `Preset`, `BackendDevice`, `Module`, `PreviewType`, `SampleMethod`, `Scheduler`
- Enum variants are PascalCase: `StableDiffusion1_5`, `Flux1Schnell`, `CPU`, `CUDA0`
- Struct names are PascalCase: `ModelConfig`, `Config`, `LoraSpec`, `HiresParams`, `Progress`
- Type aliases are PascalCase: `ConfigsBuilder`, `ModifierFunction`, `Configs`, `EmbeddingsStorage`, `LoraStorage`

**Builder Patterns:**
- Builder structs derive from derive_builder crate: `ModelConfigBuilder`, `ConfigBuilder`, `PresetBuilder`
- Builder fields use the same name as the struct field being built
- Builder setter methods match the field name: `.model()`, `.prompt()`, `.batch_count()`

## Code Style

**Formatting:**
- Standard Rust formatting (implied from codebase, no .rustfmt.toml or clippy.toml present)
- Follows Rust Edition 2024 as specified in Cargo.toml
- Imports are organized with `use` statements grouped logically
- Long module paths are properly nested and grouped

**Linting:**
- No explicit Clippy or rustfmt configuration file (uses Rust defaults)
- Derives appropriate traits on structs: `Debug`, `Clone`, `Default`, `Copy`, `Hash`, `PartialEq`, `Eq`
- Uses `#[non_exhaustive]` on enums for API stability: `BackendDevice`, `Module`, `DiffusionError`

## Import Organization

**Order:**
1. Standard library imports (`std::*`)
2. Third-party crate imports (chrono, derive_builder, hf_hub, image, libc, etc.)
3. Local crate imports (`crate::api`, `crate::preset`, etc.)

**Pattern from `api.rs`:**
```rust
use std::collections::HashMap;
use std::ffi::CString;
use std::ffi::c_char;
use std::path::Path;
use std::ptr::null;

use chrono::Local;
use derive_builder::Builder;
use diffusion_rs_sys::*;
use image::*;

use crate::preset::ConfigsBuilder;
```

**Path Aliases:**
- No explicit path aliases configured in code
- Modules are imported directly with relative paths via `crate::`

## Error Handling

**Patterns:**
- Uses `thiserror` crate for error definitions: `#[derive(Error, Debug)]`
- `DiffusionError` enum with variants: `Forward`, `StoreImages`, `Io`, `Upscaler`
- Builder errors use `ConfigBuilderError` from derive_builder
- Results are propagated with `?` operator: `repo.get(file)`, `image::open(&config.init_img)?`
- API errors from hf_hub are wrapped with `#[from]` in Result types
- Validation is done in builder `validate()` methods that return `Result<(), ConfigBuilderError>`

**Example from `api.rs`:**
```rust
#[non_exhaustive]
#[derive(Error, Debug)]
pub enum DiffusionError {
    #[error("The underling stablediffusion.cpp function returned NULL")]
    Forward,
    #[error(transparent)]
    StoreImages(#[from] ImageError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("The underling upscaler model returned a NULL image")]
    Upscaler,
}
```

## Logging

**Framework:** No explicit logging framework (console output only for status)

**Patterns:**
- Uses `println!` and `eprintln!` implicitly (not observed in main library code)
- Progress callbacks passed via `Sender<Progress>` for non-blocking updates
- Callback functions receive step, steps, and time information

## Comments

**When to Comment:**
- Doc comments (///) appear on public functions and types with detailed explanations
- Explain WHY code exists, not WHAT it does
- Implementation-critical comments inline with unsafe code blocks
- Comments explain memory management requirements (e.g., "This is required to support img2img after text2img generation")

**JSDoc/TSDoc:**
- Uses Rust doc comments (///) with markdown formatting
- Doc links with `[type_name]` or `crate::module::name` syntax
- Example from `modifier.rs`: `/// Add the <https://huggingface.co/ximso/RealESRGAN_x4plus_anime_6B> upscaler`

## Function Design

**Size:**
- Most public API functions are 1-3 lines (delegates to internal functions)
- Builder setter methods are single-line or contain validation logic
- Internal functions `gen_img_maybe_progress()` can be 280+ lines for complex logic

**Parameters:**
- Builders accept `Into<T>` via `setter(into)` attribute to allow flexible parameter types
- Functions take references for large types: `&Config`, `&mut ModelConfig`
- Mutable references used when state needs updating: `&mut builder`, `&mut Self`

**Return Values:**
- Public APIs return `Result<(), ErrorType>` for operations with side effects
- Builders return `Self` (via `&mut Self`) for method chaining
- Functions returning tuples for grouped data: `(ConfigBuilder, ModelConfigBuilder)`

## Module Design

**Exports:**
- Top-level re-exports in `lib.rs` with doc comments
- Public modules marked with `pub mod`
- Private modules marked with `pub(crate) mod` for internal-only: `preset_builder`

**Pattern from `lib.rs`:**
```rust
/// Safer wrapper around stable-diffusion.cpp bindings
pub mod api;

/// Presets that automatically download models from <https://huggingface.co/>
pub mod preset;

/// Add additional resources to [preset::Preset]
pub mod modifier;
pub(crate) mod preset_builder;

/// Util module
pub mod util;
```

**Barrel Files:**
- Types are re-exported from modules (e.g., `pub use diffusion_rs_sys::sample_method_t as SampleMethod`)
- Uses `pub use` to re-expose types with different names or from internal modules
- No single barrel module pattern; each module is a clear unit

## Builder Pattern Usage

**Derive Builder Crate:**
- Uses `derive_builder` crate extensively for complex configuration objects
- Builder attributes:
  - `#[builder(setter(into, strip_option))]` - Flexible type conversion and Option stripping
  - `#[builder(default = "expression")]` - Default values for fields
  - `#[builder(setter(custom))]` - Custom setter logic for complex fields
  - `#[builder(private)]` - Private fields in builder
  - `#[builder(build_fn(validate = "Self::validate"))]` - Custom validation function

**Complex Builder Example from `ModelConfig`:**
- `max_vram` field uses custom setter to build string representation
- `embeddings` uses custom setter to filter valid file extensions
- `lora_models` uses custom setter with filtering and validation
- `backend` and `params_backend` convert HashMap to string format

**Conversion Pattern:**
- `From<&ModelConfig> for ModelConfigBuilder` - Clone existing config for modification
- `TryFrom<PresetConfig> for ConfigsBuilder` - Sequential modifier application
- Both enable builder reuse and config transformation

## Unsafe Code Handling

**Patterns:**
- Wrapped in `unsafe` blocks with FFI calls to C library functions
- Comment explaining memory safety requirement
- Example: "This is required to support img2img after text2img generation otherwise the context is cached..."
- Uses `null()` and `null_mut()` for C pointer initialization
- Proper pointer casting with `as *const T`, `as *mut T`

---

*Convention analysis: 2026-06-18*
