use std::cmp::max;
use std::collections::HashMap;
use std::ffi::CString;
use std::ffi::c_char;
use std::ffi::c_void;
use std::fmt::Display;
use std::path::Path;
use std::path::PathBuf;
use std::ptr::null;
use std::ptr::null_mut;
use std::slice;

use chrono::Local;
use derive_builder::Builder;
use diffusion_rs_sys::free_upscaler_ctx;
use diffusion_rs_sys::new_upscaler_ctx;
use diffusion_rs_sys::sd_cache_mode_t;
use diffusion_rs_sys::sd_cache_params_t;
use diffusion_rs_sys::sd_ctx_params_t;
use diffusion_rs_sys::sd_embedding_t;
use diffusion_rs_sys::sd_get_default_sample_method;
use diffusion_rs_sys::sd_get_default_scheduler;
use diffusion_rs_sys::sd_guidance_params_t;
use diffusion_rs_sys::sd_image_t;
use diffusion_rs_sys::sd_img_gen_params_t;
use diffusion_rs_sys::sd_lora_t;
use diffusion_rs_sys::sd_pm_params_t;
use diffusion_rs_sys::sd_sample_params_t;
use diffusion_rs_sys::sd_set_preview_callback;
use diffusion_rs_sys::sd_slg_params_t;
use diffusion_rs_sys::sd_tiling_params_t;
use diffusion_rs_sys::upscaler_ctx_t;
use image::ImageBuffer;
use image::ImageError;
use image::RgbImage;
use libc::free;
use thiserror::Error;
use walkdir::DirEntry;
use walkdir::WalkDir;

use diffusion_rs_sys::free_sd_ctx;
use diffusion_rs_sys::new_sd_ctx;
use diffusion_rs_sys::sd_ctx_t;

/// Specify the range function
pub use diffusion_rs_sys::rng_type_t as RngFunction;

/// Sampling methods
pub use diffusion_rs_sys::sample_method_t as SampleMethod;

/// Denoiser sigma schedule
pub use diffusion_rs_sys::scheduler_t as Scheduler;

/// Prediction override
pub use diffusion_rs_sys::prediction_t as Prediction;

/// Weight type
pub use diffusion_rs_sys::sd_type_t as WeightType;

/// Preview mode
pub use diffusion_rs_sys::preview_t as PreviewType;

/// Lora mode
pub use diffusion_rs_sys::lora_apply_mode_t as LoraModeType;

static VALID_EXT: [&str; 3] = ["gguf", "safetensors", "pt"];

#[non_exhaustive]
#[derive(Error, Debug)]
/// Error that can occurs while forwarding models
pub enum DiffusionError {
    #[error("The underling stablediffusion.cpp function returned NULL")]
    Forward,
    #[error(transparent)]
    StoreImages(#[from] ImageError),
    #[error("The underling upscaler model returned a NULL image")]
    Upscaler,
}

#[repr(i32)]
#[non_exhaustive]
#[derive(Debug, Default, Copy, Clone, Hash, PartialEq, Eq)]
/// Ignore the lower X layers of CLIP network
pub enum ClipSkip {
    /// Will be [ClipSkip::None] for SD1.x, [ClipSkip::OneLayer] for SD2.x
    #[default]
    Unspecified = 0,
    None = 1,
    OneLayer = 2,
}

type EmbeddingsStorage = (PathBuf, Vec<(CLibString, CLibPath)>, Vec<sd_embedding_t>);

#[derive(Default, Debug, Clone)]
struct LoraStorage {
    lora_model_dir: CLibPath,
    data: Vec<(CLibPath, String, f32)>,
    loras_t: Vec<sd_lora_t>,
}

/// Specify the instructions for a Lora model
#[derive(Default, Debug, Clone)]
pub struct LoraSpec {
    pub file_name: String,
    pub is_high_noise: bool,
    pub multiplier: f32,
}

/// Parameters for UCache
#[derive(Builder, Debug, Clone)]
pub struct UCacheParams {
    /// Error threshold for reuse decision
    #[builder(default = "1.0")]
    threshold: f32,

    /// Start caching at this percent of steps
    #[builder(default = "0.15")]
    start: f32,

    /// Stop caching at this percent of steps
    #[builder(default = "0.95")]
    end: f32,

    /// Error decay rate (0-1)
    #[builder(default = "1.0")]
    decay: f32,

    /// Scale threshold by output norm
    #[builder(default = "true")]
    relative: bool,

    /// Reset error after computing
    /// true: Resets accumulated error after each computed step. More aggressive caching, works well with most samplers.
    /// false: Keeps error accumulated. More conservative, recommended for euler_a sampler
    #[builder(default = "true")]
    reset: bool,
}

/// Parameters for Easy Cache
#[derive(Builder, Debug, Clone)]
pub struct EasyCacheParams {
    /// Error threshold for reuse decision
    #[builder(default = "0.2")]
    threshold: f32,

    /// Start caching at this percent of steps
    #[builder(default = "0.15")]
    start: f32,

    /// Stop caching at this percent of steps
    #[builder(default = "0.95")]
    end: f32,
}

/// Parameters for Db Cache
#[derive(Builder, Debug, Clone)]
pub struct DbCacheParams {
    /// Front blocks to always compute
    #[builder(default = "8")]
    fn_blocks: i32,

    /// Back blocks to always compute
    #[builder(default = "0")]
    bn_blocks: i32,

    /// L1 residual difference threshold
    #[builder(default = "0.08")]
    threshold: f32,

    /// Steps before caching starts
    #[builder(default = "8")]
    warmup: i32,

    /// Steps Computation Mask controls which steps can be cached
    scm_mask: ScmPreset,

    /// Scm Policy
    #[builder(default = "ScmPolicy::default()")]
    scm_policy_dynamic: ScmPolicy,
}

/// Steps Computation Mask Policy controls when to cache steps
#[derive(Debug, Default, Clone)]
pub enum ScmPolicy {
    /// Always cache on cacheable steps
    Static,
    #[default]
    /// Check threshold before caching
    Dynamic,
}

/// Steps Computation Mask Preset controls which steps can be cached
#[derive(Debug, Default, Clone)]
pub enum ScmPreset {
    Slow,
    #[default]
    Medium,
    Fast,
    Ultra,
    /// E.g.: "1,1,1,1,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,1"
    /// where 1 means compute, 0 means cache
    Custom(String),
}

impl ScmPreset {
    fn to_vec_string(&self, steps: i32) -> String {
        match self {
            ScmPreset::Slow => ScmPresetBins {
                compute_bins: vec![8, 3, 3, 2, 1, 1],
                cache_bins: vec![1, 2, 2, 2, 3],
                steps,
            }
            .to_string(),
            ScmPreset::Medium => ScmPresetBins {
                compute_bins: vec![6, 2, 2, 2, 2, 1],
                cache_bins: vec![1, 3, 3, 3, 3],
                steps,
            }
            .to_string(),
            ScmPreset::Fast => ScmPresetBins {
                compute_bins: vec![6, 1, 1, 1, 1, 1],
                cache_bins: vec![1, 3, 4, 5, 4],
                steps,
            }
            .to_string(),
            ScmPreset::Ultra => ScmPresetBins {
                compute_bins: vec![4, 1, 1, 1, 1],
                cache_bins: vec![2, 5, 6, 7],
                steps,
            }
            .to_string(),
            ScmPreset::Custom(s) => s.clone(),
        }
    }
}

#[derive(Debug, Clone)]
struct ScmPresetBins {
    compute_bins: Vec<i32>,
    cache_bins: Vec<i32>,
    steps: i32,
}

impl ScmPresetBins {
    fn maybe_scale(&self) -> ScmPresetBins {
        if self.steps == 28 || self.steps <= 0 {
            return self.clone();
        }
        self.scale()
    }

    fn scale(&self) -> ScmPresetBins {
        let scale = self.steps as f32 / 28.0;
        let scaled_compute_bins = self
            .compute_bins
            .iter()
            .map(|b| max(1, (*b as f32 * scale * 0.5) as i32))
            .collect();
        let scaled_cached_bins = self
            .cache_bins
            .iter()
            .map(|b| max(1, (*b as f32 * scale * 0.5) as i32))
            .collect();
        ScmPresetBins {
            compute_bins: scaled_compute_bins,
            cache_bins: scaled_cached_bins,
            steps: self.steps,
        }
    }

    fn generate_vec_mask(&self) -> Vec<i32> {
        let mut mask = Vec::new();
        let mut c_idx = 0;
        let mut cache_idx = 0;

        while mask.len() < self.steps as usize {
            if c_idx < self.compute_bins.len() {
                let compute_count = self.compute_bins[c_idx];
                for _ in 0..compute_count {
                    if mask.len() < self.steps as usize {
                        mask.push(1);
                    }
                }
                c_idx += 1;
            }
            if cache_idx < self.cache_bins.len() {
                let cache_count = self.cache_bins[c_idx];
                for _ in 0..cache_count {
                    if mask.len() < self.steps as usize {
                        mask.push(0);
                    }
                }
                cache_idx += 1;
            }
            if c_idx >= self.compute_bins.len() && cache_idx >= self.cache_bins.len() {
                break;
            }
        }
        if let Some(last) = mask.last_mut() {
            *last = 1;
        }
        mask
    }
}

impl Display for ScmPresetBins {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mask: String = self
            .maybe_scale()
            .generate_vec_mask()
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(",");
        write!(f, "{mask}")
    }
}

/// Config struct for a specific diffusion model
#[derive(Builder, Debug, Clone)]
#[builder(
    setter(into, strip_option),
    build_fn(error = "ConfigBuilderError", validate = "Self::validate")
)]
pub struct ModelConfig {
    /// Number of threads to use during computation (default: 0).
    /// If n_ threads <= 0, then threads will be set to the number of CPU physical cores.
    #[builder(default = "num_cpus::get_physical() as i32", setter(custom))]
    n_threads: i32,

    /// Whether to memory-map model (default: false)
    #[builder(default = "false")]
    enable_mmap: bool,

    /// Place the weights in RAM to save VRAM, and automatically load them into VRAM when needed
    #[builder(default = "false")]
    offload_params_to_cpu: bool,

    /// Path to esrgan model. Upscale images after generate, just RealESRGAN_x4plus_anime_6B supported by now
    #[builder(default = "Default::default()")]
    upscale_model: Option<CLibPath>,

    /// Run the ESRGAN upscaler this many times (default 1)
    #[builder(default = "1")]
    upscale_repeats: i32,

    /// Tile size for ESRGAN upscaler (default 128)
    #[builder(default = "128")]
    upscale_tile_size: i32,

    /// Path to full model
    #[builder(default = "Default::default()")]
    model: CLibPath,

    /// Path to the standalone diffusion model
    #[builder(default = "Default::default()")]
    diffusion_model: CLibPath,

    /// Path to the qwen2vl text encoder
    #[builder(default = "Default::default()")]
    llm: CLibPath,

    /// Path to the qwen2vl vit
    #[builder(default = "Default::default()")]
    llm_vision: CLibPath,

    /// Path to the clip-l text encoder
    #[builder(default = "Default::default()")]
    clip_l: CLibPath,

    /// Path to the clip-g text encoder
    #[builder(default = "Default::default()")]
    clip_g: CLibPath,

    /// Path to the clip-vision encoder
    #[builder(default = "Default::default()")]
    clip_vision: CLibPath,

    /// Path to the t5xxl text encoder
    #[builder(default = "Default::default()")]
    t5xxl: CLibPath,

    /// Path to vae
    #[builder(default = "Default::default()")]
    vae: CLibPath,

    /// Path to taesd. Using Tiny AutoEncoder for fast decoding (low quality)
    #[builder(default = "Default::default()")]
    taesd: CLibPath,

    /// Path to control net model
    #[builder(default = "Default::default()")]
    control_net: CLibPath,

    /// Path to embeddings
    #[builder(default = "Default::default()", setter(custom))]
    embeddings: EmbeddingsStorage,

    /// Path to PHOTOMAKER model
    #[builder(default = "Default::default()")]
    photo_maker: CLibPath,

    /// Path to PHOTOMAKER v2 id embed
    #[builder(default = "Default::default()")]
    pm_id_embed_path: CLibPath,

    /// Weight type. If not specified, the default is the type of the weight file
    #[builder(default = "WeightType::SD_TYPE_COUNT")]
    weight_type: WeightType,

    /// Lora model directory
    #[builder(default = "Default::default()", setter(custom))]
    lora_models: LoraStorage,

    /// Path to the standalone high noise diffusion model
    #[builder(default = "Default::default()")]
    high_noise_diffusion_model: CLibPath,

    /// Process vae in tiles to reduce memory usage (default: false)
    #[builder(default = "false")]
    vae_tiling: bool,

    /// Tile size for vae tiling (default: 32x32)
    #[builder(default = "(32,32)")]
    vae_tile_size: (i32, i32),

    /// Relative tile size for vae tiling, in fraction of image size if < 1, in number of tiles per dim if >=1 (overrides vae_tile_size)
    #[builder(default = "(0.,0.)")]
    vae_relative_tile_size: (f32, f32),

    /// Tile overlap for vae tiling, in fraction of tile size (default: 0.5)
    #[builder(default = "0.5")]
    vae_tile_overlap: f32,

    /// RNG (default: CUDA)
    #[builder(default = "RngFunction::CUDA_RNG")]
    rng: RngFunction,

    /// Sampler RNG. If [RngFunction::RNG_TYPE_COUNT] is used will default to rng value. (default: [RngFunction::RNG_TYPE_COUNT])",
    #[builder(default = "RngFunction::RNG_TYPE_COUNT")]
    sampler_rng_type: RngFunction,

    /// Denoiser sigma schedule (default: [Scheduler::SCHEDULER_COUNT]).
    /// Will default to [Scheduler::EXPONENTIAL_SCHEDULER] if a denoiser is already instantiated.
    /// Otherwise, [Scheduler::DISCRETE_SCHEDULER] is used.
    #[builder(default = "Scheduler::SCHEDULER_COUNT")]
    scheduler: Scheduler,

    /// Custom sigma values for the sampler
    #[builder(default = "Default::default()")]
    sigmas: Vec<f32>,

    /// Prediction type override (default: PREDICTION_COUNT)
    #[builder(default = "Prediction::PREDICTION_COUNT")]
    prediction: Prediction,

    /// Keep vae in cpu (for low vram) (default: false)
    #[builder(default = "false")]
    vae_on_cpu: bool,

    /// keep clip in cpu (for low vram) (default: false)
    #[builder(default = "false")]
    clip_on_cpu: bool,

    /// Keep controlnet in cpu (for low vram) (default: false)
    #[builder(default = "false")]
    control_net_cpu: bool,

    /// Use flash attention to reduce memory usage (for low vram).
    // /For most backends, it slows things down, but for cuda it generally speeds it up too. At the moment, it is only supported for some models and some backends (like cpu, cuda/rocm, metal).
    #[builder(default = "false")]
    flash_attention: bool,

    /// Disable dit mask for chroma
    #[builder(default = "false")]
    chroma_disable_dit_mask: bool,

    /// Enable t5 mask for chroma
    #[builder(default = "false")]
    chroma_enable_t5_mask: bool,

    /// t5 mask pad size of chroma
    #[builder(default = "1")]
    chroma_t5_mask_pad: i32,

    /// Use qwen image zero cond true optimization
    #[builder(default = "false")]
    use_qwen_image_zero_cond_true: bool,

    /// Use Conv2d direct in the diffusion model
    /// This might crash if it is not supported by the backend.
    #[builder(default = "false")]
    diffusion_conv_direct: bool,

    /// Use Conv2d direct in the vae model (should improve the performance)
    /// This might crash if it is not supported by the backend.
    #[builder(default = "false")]
    vae_conv_direct: bool,

    /// Force use of conv scale on sdxl vae
    #[builder(default = "false")]
    force_sdxl_vae_conv_scale: bool,

    /// Shift value for Flow models like SD3.x or WAN (default: auto)
    #[builder(default = "f32::INFINITY")]
    flow_shift: f32,

    /// Shift timestep for NitroFusion models, default: 0, recommended N for NitroSD-Realism around 250 and 500 for NitroSD-Vibrant
    #[builder(default = "0")]
    timestep_shift: i32,

    /// Prevents usage of taesd for decoding the final image
    #[builder(default = "false")]
    taesd_preview_only: bool,

    /// In auto mode, if the model weights contain any quantized parameters, the at_runtime mode will be used; otherwise, immediately will be used.The immediate mode may have precision and compatibility issues with quantized parameters, but it usually offers faster inference speed and, in some cases, lower memory usage. The at_runtime mode, on the other hand, is exactly the opposite
    #[builder(default = "LoraModeType::LORA_APPLY_AUTO")]
    lora_apply_mode: LoraModeType,

    /// Enable circular padding for convolutions
    #[builder(default = "false")]
    circular: bool,

    /// Enable circular RoPE wrapping on x-axis (width) only
    #[builder(default = "false")]
    circular_x: bool,

    /// Enable circular RoPE wrapping on y-axis (height) only
    #[builder(default = "false")]
    circular_y: bool,

    #[builder(default = "None", private)]
    upscaler_ctx: Option<*mut upscaler_ctx_t>,

    #[builder(default = "None", private)]
    diffusion_ctx: Option<(*mut sd_ctx_t, sd_ctx_params_t)>,
}

impl ModelConfigBuilder {
    fn validate(&self) -> Result<(), ConfigBuilderError> {
        self.validate_model()
    }

    fn validate_model(&self) -> Result<(), ConfigBuilderError> {
        self.model
            .as_ref()
            .or(self.diffusion_model.as_ref())
            .map(|_| ())
            .ok_or(ConfigBuilderError::UninitializedField(
                "Model OR DiffusionModel must be valorized",
            ))
    }

    fn filter_valid_extensions(&self, path: &Path) -> impl Iterator<Item = DirEntry> {
        WalkDir::new(path)
            .into_iter()
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry
                    .path()
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext_str| VALID_EXT.contains(&ext_str))
                    .unwrap_or(false)
            })
    }

    fn build_single_lora_storage(
        spec: &LoraSpec,
        is_high_noise: bool,
        valid_loras: &HashMap<String, PathBuf>,
    ) -> ((CLibPath, String, f32), sd_lora_t) {
        let path = valid_loras.get(&spec.file_name).unwrap().as_path();
        let c_path = CLibPath::from(path);
        let lora = sd_lora_t {
            is_high_noise,
            multiplier: spec.multiplier,
            path: c_path.as_ptr(),
        };
        let data = (c_path, spec.file_name.clone(), spec.multiplier);
        (data, lora)
    }

    pub fn embeddings(&mut self, embeddings_dir: &Path) -> &mut Self {
        let data: Vec<(CLibString, CLibPath)> = self
            .filter_valid_extensions(embeddings_dir)
            .map(|entry| {
                let file_stem = entry
                    .path()
                    .file_stem()
                    .and_then(|stem| stem.to_str())
                    .unwrap_or_default()
                    .to_owned();
                (CLibString::from(file_stem), CLibPath::from(entry.path()))
            })
            .collect();
        let data_pointer = data
            .iter()
            .map(|(name, path)| sd_embedding_t {
                name: name.as_ptr(),
                path: path.as_ptr(),
            })
            .collect();
        self.embeddings = Some((embeddings_dir.to_path_buf(), data, data_pointer));
        self
    }

    pub fn lora_models(&mut self, lora_model_dir: &Path, specs: Vec<LoraSpec>) -> &mut Self {
        let valid_loras: HashMap<String, PathBuf> = self
            .filter_valid_extensions(lora_model_dir)
            .map(|entry| {
                let path = entry.path();
                (
                    path.file_stem()
                        .and_then(|stem| stem.to_str())
                        .unwrap_or_default()
                        .to_owned(),
                    path.to_path_buf(),
                )
            })
            .collect();
        let valid_lora_names: Vec<&String> = valid_loras.keys().collect();
        let standard = specs
            .iter()
            .filter(|s| valid_lora_names.contains(&&s.file_name) && !s.is_high_noise)
            .map(|s| Self::build_single_lora_storage(s, false, &valid_loras));
        let high_noise = specs
            .iter()
            .filter(|s| valid_lora_names.contains(&&s.file_name) && s.is_high_noise)
            .map(|s| Self::build_single_lora_storage(s, true, &valid_loras));

        let mut data = Vec::new();
        let mut loras_t = Vec::new();
        for lora in standard.chain(high_noise) {
            data.push(lora.0);
            loras_t.push(lora.1);
        }

        self.lora_models = Some(LoraStorage {
            lora_model_dir: lora_model_dir.into(),
            data,
            loras_t,
        });
        self
    }

    pub fn n_threads(&mut self, value: i32) -> &mut Self {
        self.n_threads = if value > 0 {
            Some(value)
        } else {
            Some(num_cpus::get_physical() as i32)
        };
        self
    }
}

impl ModelConfig {
    unsafe fn upscaler_ctx(&mut self) -> Option<*mut upscaler_ctx_t> {
        unsafe {
            if self.upscale_model.is_none() || self.upscale_repeats == 0 {
                None
            } else {
                if self.upscaler_ctx.is_none() {
                    let upscaler = new_upscaler_ctx(
                        self.upscale_model.as_ref().unwrap().as_ptr(),
                        self.offload_params_to_cpu,
                        self.diffusion_conv_direct,
                        self.n_threads,
                        self.upscale_tile_size,
                    );
                    self.upscaler_ctx = Some(upscaler);
                }
                self.upscaler_ctx
            }
        }
    }

    unsafe fn diffusion_ctx(&mut self, vae_decode_only: bool) -> *mut sd_ctx_t {
        unsafe {
            if self.diffusion_ctx.is_none() {
                let sd_ctx_params = sd_ctx_params_t {
                    model_path: self.model.as_ptr(),
                    llm_path: self.llm.as_ptr(),
                    llm_vision_path: self.llm_vision.as_ptr(),
                    clip_l_path: self.clip_l.as_ptr(),
                    clip_g_path: self.clip_g.as_ptr(),
                    clip_vision_path: self.clip_vision.as_ptr(),
                    high_noise_diffusion_model_path: self.high_noise_diffusion_model.as_ptr(),
                    t5xxl_path: self.t5xxl.as_ptr(),
                    diffusion_model_path: self.diffusion_model.as_ptr(),
                    vae_path: self.vae.as_ptr(),
                    taesd_path: self.taesd.as_ptr(),
                    control_net_path: self.control_net.as_ptr(),
                    embeddings: self.embeddings.2.as_ptr(),
                    embedding_count: self.embeddings.1.len() as u32,
                    photo_maker_path: self.photo_maker.as_ptr(),
                    vae_decode_only,
                    free_params_immediately: false,
                    n_threads: self.n_threads,
                    wtype: self.weight_type,
                    rng_type: self.rng,
                    keep_clip_on_cpu: self.clip_on_cpu,
                    keep_control_net_on_cpu: self.control_net_cpu,
                    keep_vae_on_cpu: self.vae_on_cpu,
                    diffusion_flash_attn: self.flash_attention,
                    diffusion_conv_direct: self.diffusion_conv_direct,
                    chroma_use_dit_mask: !self.chroma_disable_dit_mask,
                    chroma_use_t5_mask: self.chroma_enable_t5_mask,
                    chroma_t5_mask_pad: self.chroma_t5_mask_pad,
                    vae_conv_direct: self.vae_conv_direct,
                    offload_params_to_cpu: self.offload_params_to_cpu,
                    flow_shift: self.flow_shift,
                    prediction: self.prediction,
                    force_sdxl_vae_conv_scale: self.force_sdxl_vae_conv_scale,
                    tae_preview_only: self.taesd_preview_only,
                    lora_apply_mode: self.lora_apply_mode,
                    tensor_type_rules: null_mut(),
                    sampler_rng_type: self.sampler_rng_type,
                    circular_x: self.circular || self.circular_x,
                    circular_y: self.circular || self.circular_y,
                    qwen_image_zero_cond_t: self.use_qwen_image_zero_cond_true,
                    enable_mmap: self.enable_mmap,
                };
                let ctx = new_sd_ctx(&sd_ctx_params);
                self.diffusion_ctx = Some((ctx, sd_ctx_params))
            }
            self.diffusion_ctx.unwrap().0
        }
    }
}

impl Drop for ModelConfig {
    fn drop(&mut self) {
        //Cleanup CTX section
        unsafe {
            if let Some((sd_ctx, _)) = self.diffusion_ctx {
                free_sd_ctx(sd_ctx);
            }

            if let Some(upscaler_ctx) = self.upscaler_ctx {
                free_upscaler_ctx(upscaler_ctx);
            }
        }
    }
}

impl From<ModelConfig> for ModelConfigBuilder {
    fn from(value: ModelConfig) -> Self {
        let mut builder = ModelConfigBuilder::default();
        builder
            .n_threads(value.n_threads)
            .offload_params_to_cpu(value.offload_params_to_cpu)
            .upscale_repeats(value.upscale_repeats)
            .model(value.model.clone())
            .diffusion_model(value.diffusion_model.clone())
            .llm(value.llm.clone())
            .llm_vision(value.llm_vision.clone())
            .clip_l(value.clip_l.clone())
            .clip_g(value.clip_g.clone())
            .clip_vision(value.clip_vision.clone())
            .t5xxl(value.t5xxl.clone())
            .vae(value.vae.clone())
            .taesd(value.taesd.clone())
            .control_net(value.control_net.clone())
            .embeddings(&value.embeddings.0)
            .photo_maker(value.photo_maker.clone())
            .pm_id_embed_path(value.pm_id_embed_path.clone())
            .weight_type(value.weight_type)
            .high_noise_diffusion_model(value.high_noise_diffusion_model.clone())
            .vae_tiling(value.vae_tiling)
            .vae_tile_size(value.vae_tile_size)
            .vae_relative_tile_size(value.vae_relative_tile_size)
            .vae_tile_overlap(value.vae_tile_overlap)
            .rng(value.rng)
            .sampler_rng_type(value.rng)
            .scheduler(value.scheduler)
            .sigmas(value.sigmas.clone())
            .prediction(value.prediction)
            .vae_on_cpu(value.vae_on_cpu)
            .clip_on_cpu(value.clip_on_cpu)
            .control_net(value.control_net.clone())
            .control_net_cpu(value.control_net_cpu)
            .flash_attention(value.flash_attention)
            .chroma_disable_dit_mask(value.chroma_disable_dit_mask)
            .chroma_enable_t5_mask(value.chroma_enable_t5_mask)
            .chroma_t5_mask_pad(value.chroma_t5_mask_pad)
            .diffusion_conv_direct(value.diffusion_conv_direct)
            .vae_conv_direct(value.vae_conv_direct)
            .force_sdxl_vae_conv_scale(value.force_sdxl_vae_conv_scale)
            .flow_shift(value.flow_shift)
            .timestep_shift(value.timestep_shift)
            .taesd_preview_only(value.taesd_preview_only)
            .lora_apply_mode(value.lora_apply_mode)
            .circular(value.circular)
            .circular_x(value.circular_x)
            .circular_y(value.circular_y)
            .use_qwen_image_zero_cond_true(value.use_qwen_image_zero_cond_true);

        let lora_model_dir = Into::<PathBuf>::into(&value.lora_models.lora_model_dir);
        let lora_specs = value
            .lora_models
            .data
            .iter()
            .map(|(_, name, multiplier)| LoraSpec {
                file_name: name.clone(),
                is_high_noise: false,
                multiplier: *multiplier,
            })
            .collect();

        builder.lora_models(&lora_model_dir, lora_specs);

        if let Some(model) = &value.upscale_model {
            builder.upscale_model(model.clone());
        }
        builder
    }
}

#[derive(Builder, Debug, Clone)]
#[builder(setter(into, strip_option), build_fn(validate = "Self::validate"))]
/// Config struct common to all diffusion methods
pub struct Config {
    /// Path to PHOTOMAKER input id images dir
    #[builder(default = "Default::default()")]
    pm_id_images_dir: CLibPath,

    /// Path to the input image, required by img2img
    #[builder(default = "Default::default()")]
    init_img: CLibPath,

    /// Path to image condition, control net
    #[builder(default = "Default::default()")]
    control_image: CLibPath,

    /// Path to write result image to (default: ./output.png)
    #[builder(default = "PathBuf::from(\"./output.png\")")]
    output: PathBuf,

    /// Path to write result image to (default: ./output.png)
    #[builder(default = "PathBuf::from(\"./preview_output.png\")")]
    preview_output: PathBuf,

    /// Preview method
    #[builder(default = "PreviewType::PREVIEW_NONE")]
    preview_mode: PreviewType,

    /// Enables previewing noisy inputs of the models rather than the denoised outputs
    #[builder(default = "false")]
    preview_noisy: bool,

    /// Interval in denoising steps between consecutive updates of the image preview file (default is 1, meaning updating at every step)
    #[builder(default = "1")]
    preview_interval: i32,

    /// The prompt to render
    prompt: String,

    /// The negative prompt (default: "")
    #[builder(default = "\"\".into()")]
    negative_prompt: CLibString,

    /// Unconditional guidance scale (default: 7.0)
    #[builder(default = "7.0")]
    cfg_scale: f32,

    /// Distilled guidance scale for models with guidance input (default: 3.5)
    #[builder(default = "3.5")]
    guidance: f32,

    /// Strength for noising/unnoising (default: 0.75)
    #[builder(default = "0.75")]
    strength: f32,

    /// Strength for keeping input identity (default: 20%)
    #[builder(default = "20.0")]
    pm_style_strength: f32,

    /// Strength to apply Control Net (default: 0.9)
    /// 1.0 corresponds to full destruction of information in init
    #[builder(default = "0.9")]
    control_strength: f32,

    /// Image height, in pixel space (default: 512)
    #[builder(default = "512")]
    height: i32,

    /// Image width, in pixel space (default: 512)
    #[builder(default = "512")]
    width: i32,

    /// Sampling-method (default: [SampleMethod::SAMPLE_METHOD_COUNT]).
    /// [SampleMethod::EULER_SAMPLE_METHOD] will be used for flux, sd3, wan, qwen_image.
    /// Otherwise [SampleMethod::EULER_A_SAMPLE_METHOD] is used.
    #[builder(default = "SampleMethod::SAMPLE_METHOD_COUNT")]
    sampling_method: SampleMethod,

    /// eta in DDIM, only for DDIM and TCD: (default: 0)
    #[builder(default = "0.")]
    eta: f32,

    /// Number of sample steps (default: 20)
    #[builder(default = "20")]
    steps: i32,

    /// RNG seed (default: 42, use random seed for < 0)
    #[builder(default = "42")]
    seed: i64,

    /// Number of images to generate (default: 1)
    #[builder(default = "1")]
    batch_count: i32,

    /// Ignore last layers of CLIP network; 1 ignores none, 2 ignores one layer (default: -1)
    /// <= 0 represents unspecified, will be 1 for SD1.x, 2 for SD2.x
    #[builder(default = "ClipSkip::Unspecified")]
    clip_skip: ClipSkip,

    /// Apply canny preprocessor (edge detection) (default: false)
    #[builder(default = "false")]
    canny: bool,

    /// skip layer guidance (SLG) scale, only for DiT models: (default: 0)
    /// 0 means disabled, a value of 2.5 is nice for sd3.5 medium
    #[builder(default = "0.")]
    slg_scale: f32,

    /// Layers to skip for SLG steps: (default: \[7,8,9\])
    #[builder(default = "vec![7, 8, 9]")]
    skip_layer: Vec<i32>,

    /// SLG enabling point: (default: 0.01)
    #[builder(default = "0.01")]
    skip_layer_start: f32,

    /// SLG disabling point: (default: 0.2)
    #[builder(default = "0.2")]
    skip_layer_end: f32,

    /// Disable auto resize of ref images
    #[builder(default = "false")]
    disable_auto_resize_ref_image: bool,

    #[builder(default = "Self::cache_init()", private)]
    cache: sd_cache_params_t,

    #[builder(default = "CLibString::default()", private)]
    scm_mask: CLibString,
}

impl ConfigBuilder {
    fn validate(&self) -> Result<(), ConfigBuilderError> {
        self.validate_output_dir()
    }

    fn validate_output_dir(&self) -> Result<(), ConfigBuilderError> {
        let is_dir = self.output.as_ref().is_some_and(|val| val.is_dir());
        let multiple_items = self.batch_count.as_ref().is_some_and(|val| *val > 1);
        if is_dir == multiple_items {
            Ok(())
        } else {
            Err(ConfigBuilderError::ValidationError(
                "When batch_count > 1, output should point to folder and vice versa".to_owned(),
            ))
        }
    }

    fn cache_init() -> sd_cache_params_t {
        sd_cache_params_t {
            mode: sd_cache_mode_t::SD_CACHE_DISABLED,
            reuse_threshold: 1.0,
            start_percent: 0.15,
            end_percent: 0.95,
            error_decay_rate: 1.0,
            use_relative_threshold: true,
            reset_error_on_compute: true,
            Fn_compute_blocks: 8,
            Bn_compute_blocks: 0,
            residual_diff_threshold: 0.08,
            max_warmup_steps: 8,
            max_cached_steps: -1,
            max_continuous_cached_steps: -1,
            taylorseer_n_derivatives: 1,
            taylorseer_skip_interval: 1,
            scm_mask: null(),
            scm_policy_dynamic: true,
        }
    }

    pub fn no_caching(&mut self) -> &mut Self {
        let mut cache = Self::cache_init();
        cache.mode = sd_cache_mode_t::SD_CACHE_DISABLED;
        self.cache = Some(cache);
        self
    }

    pub fn ucache_caching(&mut self, params: UCacheParams) -> &mut Self {
        let mut cache = Self::cache_init();
        cache.mode = sd_cache_mode_t::SD_CACHE_UCACHE;
        cache.reuse_threshold = params.threshold;
        cache.start_percent = params.start;
        cache.end_percent = params.end;
        cache.error_decay_rate = params.decay;
        cache.use_relative_threshold = params.relative;
        cache.reset_error_on_compute = params.reset;
        self.cache = Some(cache);
        self
    }

    pub fn easy_cache_caching(&mut self, params: EasyCacheParams) -> &mut Self {
        let mut cache = Self::cache_init();
        cache.mode = sd_cache_mode_t::SD_CACHE_EASYCACHE;
        cache.reuse_threshold = params.threshold;
        cache.start_percent = params.start;
        cache.end_percent = params.end;
        self.cache = Some(cache);
        self
    }

    pub fn db_cache_caching(&mut self, params: DbCacheParams) -> &mut Self {
        let mut cache = Self::cache_init();
        cache.mode = sd_cache_mode_t::SD_CACHE_DBCACHE;
        cache.Fn_compute_blocks = params.fn_blocks;
        cache.Bn_compute_blocks = params.bn_blocks;
        cache.residual_diff_threshold = params.threshold;
        cache.max_warmup_steps = params.warmup;
        cache.scm_policy_dynamic = match params.scm_policy_dynamic {
            ScmPolicy::Static => false,
            ScmPolicy::Dynamic => true,
        };
        self.scm_mask = Some(CLibString::from(
            params
                .scm_mask
                .to_vec_string(self.steps.unwrap_or_default()),
        ));
        cache.scm_mask = self.scm_mask.as_ref().unwrap().as_ptr();

        self.cache = Some(cache);
        self
    }

    pub fn taylor_seer_caching(&mut self) -> &mut Self {
        let mut cache = Self::cache_init();
        cache.mode = sd_cache_mode_t::SD_CACHE_TAYLORSEER;
        self.cache = Some(cache);
        self
    }

    pub fn cache_dit_caching(&mut self, params: DbCacheParams) -> &mut Self {
        self.db_cache_caching(params).cache.unwrap().mode = sd_cache_mode_t::SD_CACHE_CACHE_DIT;
        self
    }
}

impl From<Config> for ConfigBuilder {
    fn from(value: Config) -> Self {
        let mut builder = ConfigBuilder::default();
        let mut cache = value.cache;
        let scm_mask = value.scm_mask.clone();
        cache.scm_mask = scm_mask.as_ptr();
        builder
            .pm_id_images_dir(value.pm_id_images_dir)
            .init_img(value.init_img)
            .control_image(value.control_image)
            .output(value.output)
            .prompt(value.prompt)
            .negative_prompt(value.negative_prompt)
            .cfg_scale(value.cfg_scale)
            .strength(value.strength)
            .pm_style_strength(value.pm_style_strength)
            .control_strength(value.control_strength)
            .height(value.height)
            .width(value.width)
            .sampling_method(value.sampling_method)
            .steps(value.steps)
            .seed(value.seed)
            .batch_count(value.batch_count)
            .clip_skip(value.clip_skip)
            .slg_scale(value.slg_scale)
            .skip_layer(value.skip_layer)
            .skip_layer_start(value.skip_layer_start)
            .skip_layer_end(value.skip_layer_end)
            .canny(value.canny)
            .disable_auto_resize_ref_image(value.disable_auto_resize_ref_image)
            .preview_output(value.preview_output)
            .preview_mode(value.preview_mode)
            .preview_noisy(value.preview_noisy)
            .preview_interval(value.preview_interval)
            .cache(cache)
            .scm_mask(scm_mask);
        builder
    }
}

#[derive(Debug, Clone, Default)]
struct CLibString(CString);

impl CLibString {
    fn as_ptr(&self) -> *const c_char {
        self.0.as_ptr()
    }
}

impl From<&str> for CLibString {
    fn from(value: &str) -> Self {
        Self(CString::new(value).unwrap())
    }
}

impl From<String> for CLibString {
    fn from(value: String) -> Self {
        Self(CString::new(value).unwrap())
    }
}

#[derive(Debug, Clone, Default)]
struct CLibPath(CString);

impl CLibPath {
    fn as_ptr(&self) -> *const c_char {
        self.0.as_ptr()
    }
}

impl From<PathBuf> for CLibPath {
    fn from(value: PathBuf) -> Self {
        Self(CString::new(value.to_str().unwrap_or_default()).unwrap())
    }
}

impl From<&Path> for CLibPath {
    fn from(value: &Path) -> Self {
        Self(CString::new(value.to_str().unwrap_or_default()).unwrap())
    }
}

impl From<&CLibPath> for PathBuf {
    fn from(value: &CLibPath) -> Self {
        PathBuf::from(value.0.to_str().unwrap())
    }
}

fn output_files(path: &Path, batch_size: i32) -> Vec<PathBuf> {
    let date = Local::now().format("%Y.%m.%d-%H.%M.%S");
    if batch_size == 1 {
        vec![path.into()]
    } else {
        (1..=batch_size)
            .map(|id| path.join(format!("output_{date}_{id}.png")))
            .collect()
    }
}

unsafe fn upscale(
    upscale_repeats: i32,
    upscaler_ctx: Option<*mut upscaler_ctx_t>,
    data: sd_image_t,
) -> Result<sd_image_t, DiffusionError> {
    unsafe {
        match upscaler_ctx {
            Some(upscaler_ctx) => {
                let upscale_factor = 4; // unused for RealESRGAN_x4plus_anime_6B.pth
                let mut current_image = data;
                for _ in 0..upscale_repeats {
                    let upscaled_image =
                        diffusion_rs_sys::upscale(upscaler_ctx, current_image, upscale_factor);

                    if upscaled_image.data.is_null() {
                        return Err(DiffusionError::Upscaler);
                    }

                    free(current_image.data as *mut c_void);
                    current_image = upscaled_image;
                }
                Ok(current_image)
            }
            None => Ok(data),
        }
    }
}

/// Generate an image with a prompt
pub fn gen_img(config: &Config, model_config: &mut ModelConfig) -> Result<(), DiffusionError> {
    let prompt: CLibString = CLibString::from(config.prompt.as_str());
    let files = output_files(&config.output, config.batch_count);
    unsafe {
        let sd_ctx = model_config.diffusion_ctx(true);
        let upscaler_ctx = model_config.upscaler_ctx();
        let init_image = sd_image_t {
            width: 0,
            height: 0,
            channel: 3,
            data: null_mut(),
        };
        let mask_image = sd_image_t {
            width: config.width as u32,
            height: config.height as u32,
            channel: 1,
            data: null_mut(),
        };
        let mut layers = config.skip_layer.clone();
        let guidance = sd_guidance_params_t {
            txt_cfg: config.cfg_scale,
            img_cfg: config.cfg_scale,
            distilled_guidance: config.guidance,
            slg: sd_slg_params_t {
                layers: layers.as_mut_ptr(),
                layer_count: config.skip_layer.len(),
                layer_start: config.skip_layer_start,
                layer_end: config.skip_layer_end,
                scale: config.slg_scale,
            },
        };
        let scheduler = if model_config.scheduler == Scheduler::SCHEDULER_COUNT {
            sd_get_default_scheduler(sd_ctx, config.sampling_method)
        } else {
            model_config.scheduler
        };
        let sample_method = if config.sampling_method == SampleMethod::SAMPLE_METHOD_COUNT {
            sd_get_default_sample_method(sd_ctx)
        } else {
            config.sampling_method
        };
        let sample_params = sd_sample_params_t {
            guidance,
            sample_method,
            sample_steps: config.steps,
            eta: config.eta,
            scheduler,
            shifted_timestep: model_config.timestep_shift,
            custom_sigmas: model_config.sigmas.as_mut_ptr(),
            custom_sigmas_count: model_config.sigmas.len() as i32,
        };
        let control_image = sd_image_t {
            width: 0,
            height: 0,
            channel: 3,
            data: null_mut(),
        };
        let vae_tiling_params = sd_tiling_params_t {
            enabled: model_config.vae_tiling,
            tile_size_x: model_config.vae_tile_size.0,
            tile_size_y: model_config.vae_tile_size.1,
            target_overlap: model_config.vae_tile_overlap,
            rel_size_x: model_config.vae_relative_tile_size.0,
            rel_size_y: model_config.vae_relative_tile_size.1,
        };
        let pm_params = sd_pm_params_t {
            id_images: null_mut(),
            id_images_count: 0,
            id_embed_path: model_config.pm_id_embed_path.as_ptr(),
            style_strength: config.pm_style_strength,
        };

        unsafe extern "C" fn save_preview_local(
            _step: ::std::os::raw::c_int,
            _frame_count: ::std::os::raw::c_int,
            frames: *mut sd_image_t,
            _is_noisy: bool,
            data: *mut ::std::os::raw::c_void,
        ) {
            unsafe {
                let path = &*data.cast::<PathBuf>();
                let _ = save_img(*frames, path);
            }
        }

        if config.preview_mode != PreviewType::PREVIEW_NONE {
            let data = &config.preview_output as *const PathBuf;

            sd_set_preview_callback(
                Some(save_preview_local),
                config.preview_mode,
                config.preview_interval,
                !config.preview_noisy,
                config.preview_noisy,
                data as *mut c_void,
            );
        }

        let sd_img_gen_params = sd_img_gen_params_t {
            prompt: prompt.as_ptr(),
            negative_prompt: config.negative_prompt.as_ptr(),
            clip_skip: config.clip_skip as i32,
            init_image,
            ref_images: null_mut(),
            ref_images_count: 0,
            increase_ref_index: false,
            mask_image,
            width: config.width,
            height: config.height,
            sample_params,
            strength: config.strength,
            seed: config.seed,
            batch_count: config.batch_count,
            control_image,
            control_strength: config.control_strength,
            pm_params,
            vae_tiling_params,
            auto_resize_ref_image: config.disable_auto_resize_ref_image,
            cache: config.cache,
            loras: model_config.lora_models.loras_t.as_ptr(),
            lora_count: model_config.lora_models.loras_t.len() as u32,
        };
        let slice = diffusion_rs_sys::generate_image(sd_ctx, &sd_img_gen_params);
        let ret = {
            if slice.is_null() {
                return Err(DiffusionError::Forward);
            }
            for (img, path) in slice::from_raw_parts(slice, config.batch_count as usize)
                .iter()
                .zip(files)
            {
                match upscale(model_config.upscale_repeats, upscaler_ctx, *img) {
                    Ok(img) => save_img(img, &path)?,
                    Err(err) => {
                        return Err(err);
                    }
                }
            }
            Ok(())
        };
        free(slice as *mut c_void);
        ret
    }
}

fn save_img(img: sd_image_t, path: &Path) -> Result<(), DiffusionError> {
    // Thx @wandbrandon
    let len = (img.width * img.height * img.channel) as usize;
    let buffer = unsafe { slice::from_raw_parts(img.data, len).to_vec() };
    let save_state = ImageBuffer::from_raw(img.width, img.height, buffer)
        .map(|img| RgbImage::from(img).save(path));
    if let Some(Err(err)) = save_state {
        return Err(DiffusionError::StoreImages(err));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::{
        api::{ConfigBuilderError, ModelConfigBuilder},
        util::download_file_hf_hub,
    };

    use super::{ConfigBuilder, gen_img};

    #[test]
    fn test_required_args_txt2img() {
        assert!(ConfigBuilder::default().build().is_err());
        assert!(ModelConfigBuilder::default().build().is_err());
        ModelConfigBuilder::default()
            .model(PathBuf::from("./test.ckpt"))
            .build()
            .unwrap();

        ConfigBuilder::default()
            .prompt("a lovely cat driving a sport car")
            .build()
            .unwrap();

        assert!(matches!(
            ConfigBuilder::default()
                .prompt("a lovely cat driving a sport car")
                .batch_count(10)
                .build(),
            Err(ConfigBuilderError::ValidationError(_))
        ));

        ConfigBuilder::default()
            .prompt("a lovely cat driving a sport car")
            .build()
            .unwrap();

        ConfigBuilder::default()
            .prompt("a lovely duck drinking water from a bottle")
            .batch_count(2)
            .output(PathBuf::from("./"))
            .build()
            .unwrap();
    }

    #[ignore]
    #[test]
    fn test_img_gen() {
        let model_path =
            download_file_hf_hub("CompVis/stable-diffusion-v-1-4-original", "sd-v1-4.ckpt")
                .unwrap();

        let upscaler_path = download_file_hf_hub(
            "ximso/RealESRGAN_x4plus_anime_6B",
            "RealESRGAN_x4plus_anime_6B.pth",
        )
        .unwrap();
        let config = ConfigBuilder::default()
            .prompt("a lovely duck drinking water from a bottle")
            .output(PathBuf::from("./output_1.png"))
            .batch_count(1)
            .build()
            .unwrap();
        let mut model_config = ModelConfigBuilder::default()
            .model(model_path)
            .upscale_model(upscaler_path)
            .upscale_repeats(1)
            .build()
            .unwrap();

        gen_img(&config, &mut model_config).unwrap();
        let config2 = ConfigBuilder::from(config.clone())
            .prompt("a lovely duck drinking water from a straw")
            .output(PathBuf::from("./output_2.png"))
            .build()
            .unwrap();
        gen_img(&config2, &mut model_config).unwrap();

        let config3 = ConfigBuilder::from(config)
            .prompt("a lovely dog drinking water from a starbucks cup")
            .batch_count(2)
            .output(PathBuf::from("./"))
            .build()
            .unwrap();

        gen_img(&config3, &mut model_config).unwrap();
    }
}
