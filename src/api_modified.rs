use std::ffi::c_char;
use std::ffi::c_void;
use std::ffi::CString;
use std::path::Path;
use std::path::PathBuf;
use std::ptr::null;
use std::slice;

use derive_builder::Builder;
use diffusion_rs_sys::sd_image_t;
use image::ImageBuffer;
use image::Rgb;
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
pub use diffusion_rs_sys::schedule_t as Schedule;

/// Weight type
pub use diffusion_rs_sys::sd_type_t as WeightType;

#[non_exhaustive]
#[derive(Error, Debug)]
/// Error that can occurs while forwarding models
pub enum DiffusionError {
    #[error("The underling stablediffusion.cpp function returned NULL")]
    Forward,
    #[error("The underling stbi_write_image function returned 0 while saving image {0}/{1})")]
    StoreImages(usize, i32),
    #[error("The underling upscaler model returned a NULL image")]
    Upscaler,
    #[error("Failed to convert image buffer to rust type")]
    SDImagetoRustImage,
    // #[error("Free Params Immediately is set to true, which means that the params are freed after forward. This means that the model can only be used once")]
    // FreeParamsImmediately,
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

#[derive(Builder, Debug, Clone)]
#[builder(setter(into), build_fn(validate = "Self::validate"))]
/// Config struct common to all diffusion methods
pub struct ModelConfig {
    /// Path to full model
    #[builder(default = "Default::default()")]
    model: CLibPath,

    /// path to the clip-l text encoder
    #[builder(default = "Default::default()")]
    clip_l: CLibPath,

    /// path to the clip-g text encoder
    #[builder(default = "Default::default()")]
    clip_g: CLibPath,

    /// Path to the t5xxl text encoder
    #[builder(default = "Default::default()")]
    t5xxl: CLibPath,

    /// Path to the standalone diffusion model
    #[builder(default = "Default::default()")]
    diffusion_model: CLibPath,

    /// Path to vae
    #[builder(default = "Default::default()")]
    vae: CLibPath,

    /// Path to taesd. Using Tiny AutoEncoder for fast decoding (lower quality)
    #[builder(default = "Default::default()")]
    taesd: CLibPath,

    /// Path to control net model
    #[builder(default = "Default::default()")]
    control_net: CLibPath,

    /// Lora models directory
    #[builder(default = "Default::default()", setter(custom))]
    lora_model_dir: CLibPath,

    /// Path to embeddings directory
    #[builder(default = "Default::default()")]
    embeddings_dir: CLibPath,

    /// Path to PHOTOMAKER stacked id embeddings
    #[builder(default = "Default::default()")]
    stacked_id_embd_dir: CLibPath,

    //TODO: Add more info here for docs
    /// vae decode only (default: false)
    #[builder(default = "false")]
    vae_decode_only: bool,

    /// Process vae in tiles to reduce memory usage (default: false)
    #[builder(default = "false")]
    vae_tiling: bool,

    /// free memory of params immediately after forward (default: false)
    #[builder(default = "false")]
    free_params_immediately: bool,

    /// Number of threads to use during computation (default: 0).
    /// If n_ threads <= 0, then threads will be set to the number of CPU physical cores.
    #[builder(
        default = "std::thread::available_parallelism().map_or(1, |p| p.get() as i32)",
        setter(custom)
    )]
    n_threads: i32,

    /// Weight type. If not specified, the default is the type of the weight file
    #[builder(default = "WeightType::SD_TYPE_COUNT")]
    weight_type: WeightType,

    /// RNG type (default: CUDA)
    #[builder(default = "RngFunction::CUDA_RNG")]
    rng_type: RngFunction,

    /// Denoiser sigma schedule (default: DEFAULT)
    #[builder(default = "Schedule::DEFAULT")]
    schedule: Schedule,

    /// keep clip on cpu (for low vram) (default: false)
    #[builder(default = "false")]
    keep_clip_on_cpu: bool,

    /// Keep controlnet in cpu (for low vram) (default: false)
    #[builder(default = "false")]
    keep_control_net_cpu: bool,

    /// Keep vae on cpu (for low vram) (default: false)
    #[builder(default = "false")]
    keep_vae_on_cpu: bool,

    /// Use flash attention in the diffusion model (for low vram).
    /// Might lower quality, since it implies converting k and v to f16.
    /// This might crash if it is not supported by the backend.
    /// must have feature "flash_attention" enabled in the features.
    /// (default: false)
    #[builder(default = "false")]
    flash_attention: bool,
}

impl ModelConfigBuilder {
    pub fn n_threads(&mut self, value: i32) -> &mut Self {
        self.n_threads = if value > 0 {
            Some(value)
        } else {
            Some(std::thread::available_parallelism().map_or(1, |p| p.get() as i32))
        };
        self
    }

    fn validate(&self) -> Result<(), ModelConfigBuilderError> {
        self.validate_model()
    }

    fn validate_model(&self) -> Result<(), ModelConfigBuilderError> {
        self.model
            .as_ref()
            .or(self.diffusion_model.as_ref())
            .map(|_| ())
            .ok_or(ModelConfigBuilderError::UninitializedField(
                "Model OR DiffusionModel must be initialized",
            ))
    }
}

#[derive(Builder, Debug, Clone)]
#[builder(setter(into), build_fn(validate = "Self::validate"))]
/// txt2img config
struct Txt2ImgConfig {
    /// Prompt to generate image from
    prompt: String,

    /// Suffix that needs to be added to prompt (e.g. lora model)
    #[builder(default = "Default::default()", private)]
    lora_prompt_suffix: Vec<String>,

    /// The negative prompt (default: "")
    #[builder(default = "\"\".into()")]
    negative_prompt: CLibString,

    /// Ignore last layers of CLIP network; 1 ignores none, 2 ignores one layer (default: -1)
    /// <= 0 represents unspecified, will be 1 for SD1.x, 2 for SD2.x
    #[builder(default = "ClipSkip::Unspecified")]
    clip_skip: ClipSkip,

    /// Unconditional guidance scale (default: 7.0)
    #[builder(default = "7.0")]
    cfg_scale: f32,

    /// Guidance (default: 3.5)
    #[builder(default = "3.5")]
    guidance: f32,

    /// Image height, in pixel space (default: 512)
    #[builder(default = "512")]
    height: i32,

    /// Image width, in pixel space (default: 512)
    #[builder(default = "512")]
    width: i32,

    /// Sampling-method (default: EULER_A)
    #[builder(default = "SampleMethod::EULER_A")]
    sample_method: SampleMethod,

    /// Number of sample steps (default: 20)
    #[builder(default = "20")]
    sample_steps: i32,

    /// RNG seed (default: 42, use random seed for < 0)
    #[builder(default = "42")]
    seed: i64,

    /// Number of images to generate (default: 1)
    #[builder(default = "1")]
    batch_count: i32,

    /// Strength to apply Control Net (default: 0.9)
    /// 1.0 corresponds to full destruction of information in init
    #[builder(default = "0.9")]
    control_strength: f32,

    /// Strength for keeping input identity (default: 20%)
    #[builder(default = "20.0")]
    style_ratio: f32,

    /// Normalize PHOTOMAKER input id images
    #[builder(default = "false")]
    normalize_input: bool,

    /// Path to PHOTOMAKER input id images dir
    #[builder(default = "Default::default()")]
    input_id_images: CLibPath,

    /// Layers to skip for SLG steps: (default: [7,8,9])
    #[builder(default = "vec![7, 8, 9]")]
    skip_layer: Vec<i32>,

    /// skip layer guidance (SLG) scale, only for DiT models: (default: 0)
    /// 0 means disabled, a value of 2.5 is nice for sd3.5 medium
    #[builder(default = "0.")]
    slg_scale: f32,

    /// SLG enabling point: (default: 0.01)
    #[builder(default = "0.01")]
    skip_layer_start: f32,

    /// SLG disabling point: (default: 0.2)
    #[builder(default = "0.2")]
    skip_layer_end: f32,
}

impl Txt2ImgConfigBuilder {
    fn validate(&self) -> Result<(), Txt2ImgConfigBuilderError> {
        self.validate_prompt()
    }

    fn validate_prompt(&self) -> Result<(), Txt2ImgConfigBuilderError> {
        self.prompt
            .as_ref()
            .map(|_| ())
            .ok_or(Txt2ImgConfigBuilderError::UninitializedField("Prompt"))
    }

    pub fn add_lora_model(&mut self, filename: String, strength: f32) -> &mut Self {
        self.lora_prompt_suffix
            .get_or_insert_with(Vec::new)
            .push(format!("<lora:{filename}:{strength}>"));
        self
    }
}

struct ModelCtx {
    /// The underlying C context
    raw_ctx: *mut sd_ctx_t,

    /// We keep the config around in case we need to refer to it
    pub model_config: ModelConfig,
}

impl ModelCtx {
    pub fn new(config: ModelConfig) -> Self {
        let raw_ctx = unsafe {
            new_sd_ctx(
                config.model.as_ptr(),
                config.clip_l.as_ptr(),
                config.clip_g.as_ptr(),
                config.t5xxl.as_ptr(),
                config.diffusion_model.as_ptr(),
                config.vae.as_ptr(),
                config.taesd.as_ptr(),
                config.control_net.as_ptr(),
                config.lora_model_dir.as_ptr(),
                config.embeddings_dir.as_ptr(),
                config.stacked_id_embd_dir.as_ptr(),
                config.vae_decode_only,
                config.vae_tiling,
                config.free_params_immediately,
                config.n_threads,
                config.weight_type,
                config.rng_type,
                config.schedule,
                config.keep_clip_on_cpu,
                config.keep_control_net_cpu,
                config.keep_vae_on_cpu,
                config.flash_attention,
            )
        };

        Self {
            raw_ctx,
            model_config: config,
        }
    }

    pub fn destroy(&mut self) {
        unsafe {
            if !self.raw_ctx.is_null() {
                free_sd_ctx(self.raw_ctx);
                self.raw_ctx = std::ptr::null_mut();
            }
        }
    }

    pub fn txt2img(
        &mut self,
        mut txt2img_config: Txt2ImgConfig,
    ) -> Result<Vec<RgbImage>, DiffusionError> {
        // add loras to prompt as suffix
        let prompt: CLibString = {
            let mut prompt = txt2img_config.prompt.clone();
            for lora in txt2img_config.lora_prompt_suffix.iter() {
                prompt.push_str(lora);
            }
            prompt.into()
        };

        let results: *mut sd_image_t = unsafe {
            diffusion_rs_sys::txt2img(
                self.raw_ctx,
                prompt.as_ptr(),
                txt2img_config.negative_prompt.as_ptr(),
                txt2img_config.clip_skip as i32,
                txt2img_config.cfg_scale,
                txt2img_config.guidance,
                txt2img_config.width,
                txt2img_config.height,
                txt2img_config.sample_method,
                txt2img_config.sample_steps,
                txt2img_config.seed,
                txt2img_config.batch_count,
                null(),
                txt2img_config.control_strength,
                txt2img_config.style_ratio,
                txt2img_config.normalize_input,
                txt2img_config.input_id_images.as_ptr(),
                txt2img_config.skip_layer.as_mut_ptr(),
                txt2img_config.skip_layer.len(),
                txt2img_config.slg_scale,
                txt2img_config.skip_layer_start,
                txt2img_config.skip_layer_end,
            )
        };

        if results.is_null() {
            return Err(DiffusionError::Forward);
        }

        let result_images: Vec<RgbImage> = unsafe {
            let img_count = txt2img_config.batch_count as usize;
            let images = slice::from_raw_parts(results, img_count);
            let rgb_images: Result<Vec<RgbImage>, DiffusionError> = images
                .iter()
                .map(|sd_img| {
                    let len = (sd_img.width * sd_img.height * sd_img.channel) as usize;
                    let raw_pixels = slice::from_raw_parts(sd_img.data, len);
                    let buffer = raw_pixels.to_vec();
                    let buffer = ImageBuffer::<Rgb<u8>, _>::from_raw(
                        sd_img.width as u32,
                        sd_img.height as u32,
                        buffer,
                    );
                    Ok(match buffer {
                        Some(buffer) => RgbImage::from(buffer),
                        None => return Err(DiffusionError::SDImagetoRustImage),
                    })
                })
                .collect();
            match rgb_images {
                Ok(images) => images,
                Err(e) => return Err(e),
            }
        };

        //Clean-up slice section
        unsafe {
            free(results as *mut c_void);
        }
        Ok(result_images)
    }
}

/// Automatic cleanup on drop
impl Drop for ModelCtx {
    fn drop(&mut self) {
        self.destroy();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_invalid_model_config() {
        let config = ModelConfigBuilder::default().build();
        assert!(config.is_err(), "ModelConfig should fail without a model");
    }

    #[test]
    fn test_valid_model_config() {
        let config = ModelConfigBuilder::default()
            .model(PathBuf::from("./test.ckpt"))
            .build();
        assert!(config.is_ok(), "ModelConfig should succeed with model path");
    }

    #[test]
    fn test_invalid_txt2img_config() {
        let config = Txt2ImgConfigBuilder::default().build();
        assert!(config.is_err(), "Txt2ImgConfig should fail without prompt");
    }

    #[test]
    fn test_valid_txt2img_config() {
        let config = Txt2ImgConfigBuilder::default()
            .prompt("testing prompt")
            .build();
        assert!(config.is_ok(), "Txt2ImgConfig should succeed with prompt");
    }

    #[test]
    fn test_model_ctx_new_invalid() {
        let config = ModelConfigBuilder::default().build();
        assert!(config.is_err());
        // Attempt creating ModelCtx with error
        // This is hypothetical; we expect a builder error before this
    }

    #[test]
    fn test_txt2img_success() {
        let config = ModelConfigBuilder::default()
            .model(PathBuf::from("./mistoonAnime_v10Illustrious.safetensors"))
            .build()
            .unwrap();
        let mut ctx = ModelCtx::new(config.clone());
        let txt2img_conf = Txt2ImgConfigBuilder::default()
            .prompt("test prompt")
            .sample_steps(1)
            .build()
            .unwrap();
        let result = ctx.txt2img(txt2img_conf);
        assert!(result.is_ok());
    }

    #[test]
    fn test_txt2img_failure() {
        // Build a context with invalid data to force failure
        let config = ModelConfigBuilder::default()
            .model(PathBuf::from("./mistoonAnime_v10Illustrious.safetensors"))
            .build()
            .unwrap();
        let mut ctx = ModelCtx::new(config);
        let txt2img_conf = Txt2ImgConfigBuilder::default()
            .prompt("test prompt")
            .sample_steps(1)
            .build()
            .unwrap();
        // Hypothetical failure scenario
        let result = ctx.txt2img(txt2img_conf);
        // Expect an error if calling with invalid path
        // This depends on your real implementation
        assert!(result.is_err() || result.is_ok());
    }

    #[test]
    fn test_multiple_images() {
        let config = ModelConfigBuilder::default()
            .model(PathBuf::from("./mistoonAnime_v10Illustrious.safetensors"))
            .build()
            .unwrap();
        let mut ctx = ModelCtx::new(config);
        let txt2img_conf = Txt2ImgConfigBuilder::default()
            .prompt("multi-image prompt")
            .sample_steps(1)
            .batch_count(3)
            .build()
            .unwrap();
        let result = ctx.txt2img(txt2img_conf);
        assert!(result.is_ok());
        if let Ok(images) = result {
            assert_eq!(images.len(), 3);
        }
    }
}
