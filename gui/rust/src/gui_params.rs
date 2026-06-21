/// FRB-compatible DTO carrying all generation parameters across the FFI boundary.
/// All fields use primitive types only (String, i32, i64, f32, bool, Option<T>)
/// to satisfy flutter_rust_bridge serialization requirements (D-11 / FRB-04).
#[derive(Debug, Clone)]
pub struct GuiParams {
    /// Preset name (must match a PresetDiscriminants variant)
    pub preset: String,
    /// Weight type name (None for presets without weight variants)
    pub weight: Option<String>,
    /// Text prompt for image generation
    pub prompt: String,
    /// Negative prompt (optional)
    pub negative_prompt: Option<String>,
    /// Number of inference steps (None uses preset default)
    pub steps: Option<i32>,
    /// Image width (None uses preset default)
    pub width: Option<i32>,
    /// Image height (None uses preset default)
    pub height: Option<i32>,
    /// Number of images to generate in a batch
    pub batch_count: i32,
    /// RNG seed (-1 for random)
    pub seed: i64,
    /// Cache acceleration mode name (None for no caching)
    pub cache_mode: Option<String>,
    /// Preview mode: "None", "Fast", or "Accurate"
    pub preview_mode: String,
    /// Upscaler mode name (None for no upscaling)
    pub upscaler: Option<String>,
    /// Upscaler scale factor
    pub upscaler_scale: f32,
    /// HuggingFace API token (optional)
    pub token: Option<String>,
    /// Enable low-VRAM optimizations (vae_tiling + flash_attention)
    pub low_vram: bool,
    /// Temp directory path for preview PNG written by the C callback
    pub preview_output: String,
    /// Output path for the final generated image
    pub output: String,
}
