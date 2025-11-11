use std::cell::RefCell;
use std::ffi::CString;
use std::ffi::c_char;
use std::ffi::c_void;
use std::path::Path;
use std::path::PathBuf;
use std::ptr::null_mut;
use std::slice;
use std::thread::spawn;

use derive_builder::Builder;
use diffusion_rs_sys::free_upscaler_ctx;
use diffusion_rs_sys::new_upscaler_ctx;
use diffusion_rs_sys::sd_ctx_params_t;
use diffusion_rs_sys::sd_guidance_params_t;
use diffusion_rs_sys::sd_image_t;
use diffusion_rs_sys::sd_img_gen_params_t;
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

    /// Place the weights in RAM to save VRAM, and automatically load them into VRAM when needed
    #[builder(default = "false")]
    offload_params_to_cpu: bool,

    /// Path to esrgan model. Upscale images after generate, just RealESRGAN_x4plus_anime_6B supported by now
    #[builder(default = "Default::default()")]
    upscale_model: Option<CLibPath>,

    /// Run the ESRGAN upscaler this many times (default 1)
    #[builder(default = "0")]
    upscale_repeats: i32,

    /// Path to full model
    #[builder(default = "Default::default()")]
    model: CLibPath,

    /// Path to the standalone diffusion model
    #[builder(default = "Default::default()")]
    diffusion_model: CLibPath,

    /// Path to the qwen2vl text encoder
    #[builder(default = "Default::default()")]
    qwen2vl: CLibPath,

    /// Path to the qwen2vl vit
    #[builder(default = "Default::default()")]
    qwen2vl_vision: CLibPath,

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
    #[builder(default = "Default::default()")]
    embeddings: CLibPath,

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
    lora_model: CLibPath,

    /// Path to the standalone high noise diffusion model
    #[builder(default = "Default::default()")]
    high_noise_diffusion_model: CLibPath,

    /// Suffix that needs to be added to prompt (e.g. lora model)
    #[builder(default = "None", private)]
    prompt_suffix: Option<String>,

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

    /// Denoiser sigma schedule (default: DEFAULT)
    #[builder(default = "Scheduler::DEFAULT")]
    scheduler: Scheduler,

    /// Prediction type override (default: DEFAULT_PRED)
    #[builder(default = "Prediction::DEFAULT_PRED")]
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

    /// Use flash attention in the diffusion model (for low vram).
    /// Might lower quality, since it implies converting k and v to f16.
    /// This might crash if it is not supported by the backend.
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

    pub fn lora_model(&mut self, lora_model: &Path) -> &mut Self {
        let folder = lora_model.parent().unwrap();
        let file_name = lora_model.file_stem().unwrap().to_str().unwrap().to_owned();
        self.prompt_suffix(format!("<lora:{file_name}:1>"));
        self.lora_model = Some(folder.into());
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
                let upscaler = new_upscaler_ctx(
                    self.upscale_model.as_ref().unwrap().as_ptr(),
                    self.offload_params_to_cpu,
                    self.diffusion_conv_direct,
                    self.n_threads,
                );
                Some(upscaler)
            }
        }
    }

    unsafe fn diffusion_ctx(&mut self, vae_decode_only: bool) -> *mut sd_ctx_t {
        unsafe {
            let sd_ctx_params = sd_ctx_params_t {
                model_path: self.model.as_ptr(),
                qwen2vl_path: self.qwen2vl.as_ptr(),
                qwen2vl_vision_path: self.qwen2vl_vision.as_ptr(),
                clip_l_path: self.clip_l.as_ptr(),
                clip_g_path: self.clip_g.as_ptr(),
                clip_vision_path: self.clip_vision.as_ptr(),
                high_noise_diffusion_model_path: self.high_noise_diffusion_model.as_ptr(),
                t5xxl_path: self.t5xxl.as_ptr(),
                diffusion_model_path: self.diffusion_model.as_ptr(),
                vae_path: self.vae.as_ptr(),
                taesd_path: self.taesd.as_ptr(),
                control_net_path: self.control_net.as_ptr(),
                lora_model_dir: self.lora_model.as_ptr(),
                embedding_dir: self.embeddings.as_ptr(),
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
            };
            new_sd_ctx(&sd_ctx_params)
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
            .qwen2vl(value.qwen2vl.clone())
            .qwen2vl_vision(value.qwen2vl_vision.clone())
            .clip_l(value.clip_l.clone())
            .clip_g(value.clip_g.clone())
            .clip_vision(value.clip_vision.clone())
            .t5xxl(value.t5xxl.clone())
            .vae(value.vae.clone())
            .taesd(value.taesd.clone())
            .control_net(value.control_net.clone())
            .embeddings(value.embeddings.clone())
            .photo_maker(value.photo_maker.clone())
            .pm_id_embed_path(value.pm_id_embed_path.clone())
            .weight_type(value.weight_type)
            .high_noise_diffusion_model(value.high_noise_diffusion_model.clone())
            .vae_tiling(value.vae_tiling)
            .vae_tile_size(value.vae_tile_size)
            .vae_relative_tile_size(value.vae_relative_tile_size)
            .vae_tile_overlap(value.vae_tile_overlap)
            .rng(value.rng)
            .scheduler(value.scheduler)
            .prediction(value.prediction)
            .vae_on_cpu(value.vae_on_cpu)
            .clip_on_cpu(value.clip_on_cpu)
            .control_net(value.control_net)
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
            .taesd_preview_only(value.taesd_preview_only);

        if let Some(model) = value.upscale_model {
            builder.upscale_model(model.clone());
        }
        
        builder.lora_model(&Into::<PathBuf>::into(&value.lora_model));

        if let Some(suffix) = value.prompt_suffix {
            builder.prompt_suffix(suffix.clone());
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

    /// Sampling-method (default: SAMPLE_METHOD_DEFAULT)
    #[builder(default = "SampleMethod::SAMPLE_METHOD_DEFAULT")]
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
}

impl From<Config> for ConfigBuilder {
    fn from(value: Config) -> Self {
        let mut builder = ConfigBuilder::default();
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
            .preview_interval(value.preview_interval);
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

fn output_files(path: &Path, prompt: &str, batch_size: i32) -> Vec<PathBuf> {
    if batch_size == 1 {
        vec![path.into()]
    } else {
        (1..=batch_size)
            .map(|id| path.join(format!("{prompt}_{id}.png")))
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
pub fn gen_img(config: Config, model_config: ModelConfig) -> Result<(), DiffusionError> {
    spawn(|| gen_img_internal(config, model_config))
        .join()
        .map_err(|_| DiffusionError::Forward)
        .flatten()
}

fn gen_img_internal(
    mut config: Config,
    mut model_config: ModelConfig,
) -> Result<(), DiffusionError> {
    let prompt: CLibString = match &model_config.prompt_suffix {
        Some(suffix) => format!("{} {suffix}", &config.prompt),
        None => config.prompt.clone(),
    }
    .into();
    let files = output_files(&config.output, &config.prompt, config.batch_count);
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
        let guidance = sd_guidance_params_t {
            txt_cfg: config.cfg_scale,
            img_cfg: config.cfg_scale,
            distilled_guidance: config.guidance,
            slg: sd_slg_params_t {
                layers: config.skip_layer.as_mut_ptr(),
                layer_count: config.skip_layer.len(),
                layer_start: config.skip_layer_start,
                layer_end: config.skip_layer_end,
                scale: config.slg_scale,
            },
        };
        let sample_params = sd_sample_params_t {
            guidance,
            sample_method: config.sampling_method,
            sample_steps: config.steps,
            eta: config.eta,
            scheduler: model_config.scheduler,
            shifted_timestep: model_config.timestep_shift,
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
        thread_local! {
            pub static LOCAL_PREVIEW_PATH: RefCell<Option<PathBuf>> = const { RefCell::new(None) };
        }

        unsafe extern "C" fn save_preview_local(
            _step: ::std::os::raw::c_int,
            _frame_count: ::std::os::raw::c_int,
            frames: *mut sd_image_t,
            _is_noisy: bool,
        ) {
            LOCAL_PREVIEW_PATH.with_borrow(|p| {
                if let Some(path) = p {
                    unsafe {
                        let _ = save_img(*frames, path);
                    };
                }
            });
        }

        if config.preview_mode != PreviewType::PREVIEW_NONE {
            LOCAL_PREVIEW_PATH.replace(Some(config.preview_output.clone()));
            sd_set_preview_callback(
                Some(save_preview_local),
                config.preview_mode,
                config.preview_interval,
                !config.preview_noisy,
                config.preview_noisy,
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
        free_sd_ctx(sd_ctx);
        if let Some(upscaler_ctx) = upscaler_ctx {
            free_upscaler_ctx(upscaler_ctx);
        }

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
        let model_config = ModelConfigBuilder::default()
            .model(model_path)
            .upscale_model(upscaler_path)
            .upscale_repeats(1)
            .build()
            .unwrap();

        gen_img(config.clone(), model_config.clone()).unwrap();
        let config2 = ConfigBuilder::from(config.clone())
            .prompt("a lovely duck drinking water from a straw")
            .output(PathBuf::from("./output_2.png"))
            .build()
            .unwrap();
        gen_img(config2, model_config.clone()).unwrap();

        let config3 = ConfigBuilder::from(config)
            .prompt("a lovely dog drinking water from a starbucks cup")
            .batch_count(2)
            .output(PathBuf::from("./"))
            .build()
            .unwrap();

        gen_img(config3, model_config).unwrap();
    }
}
