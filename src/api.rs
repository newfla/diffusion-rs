use std::ffi::c_char;
use std::ffi::c_void;
use std::ffi::CString;
use std::path::Path;
use std::path::PathBuf;
use std::ptr::null;
use std::slice;

use derive_builder::Builder;
use diffusion_rs_sys::free_upscaler_ctx;
use diffusion_rs_sys::new_upscaler_ctx;
use diffusion_rs_sys::sd_image_t;
use diffusion_rs_sys::upscaler_ctx_t;
use libc::free;
use thiserror::Error;

use diffusion_rs_sys::free_sd_ctx;
use diffusion_rs_sys::new_sd_ctx;
use diffusion_rs_sys::sd_ctx_t;
use diffusion_rs_sys::stbi_write_png_custom;

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
    #[error("The underling upsclaer model returned a NULL image")]
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

    /// path to the clip-l text encoder
    #[builder(default = "Default::default()")]
    clip_l: CLibPath,

    /// path to the clip-g text encoder
    #[builder(default = "Default::default()")]
    clip_g: CLibPath,

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

    /// Path to PHOTOMAKER stacked id embeddings
    #[builder(default = "Default::default()")]
    stacked_id_embd: CLibPath,

    /// Weight type. If not specified, the default is the type of the weight file
    #[builder(default = "WeightType::SD_TYPE_COUNT")]
    weight_type: WeightType,

    /// Lora model directory
    #[builder(default = "Default::default()", setter(custom))]
    lora_model: CLibPath,

    /// Suffix that needs to be added to prompt (e.g. lora model)
    #[builder(default = "None", private)]
    prompt_suffix: Option<String>,

    /// Process vae in tiles to reduce memory usage (default: false)
    #[builder(default = "false")]
    vae_tiling: bool,

    /// RNG (default: CUDA)
    #[builder(default = "RngFunction::CUDA_RNG")]
    rng: RngFunction,

    /// Denoiser sigma schedule (default: DEFAULT)
    #[builder(default = "Schedule::DEFAULT")]
    schedule: Schedule,

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

    #[builder(default = "None", private)]
    upscaler_ctx: Option<*mut upscaler_ctx_t>,

    #[builder(default = "None", private)]
    diffusion_ctx: Option<*mut sd_ctx_t>,
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
        if self.upscale_model.is_none() || self.upscale_repeats == 0 {
            None
        } else {
            if self.upscaler_ctx.is_none() {
                let upscaler = new_upscaler_ctx(
                    self.upscale_model.as_ref().unwrap().as_ptr(),
                    self.n_threads,
                );
                self.upscaler_ctx = Some(upscaler);
            }
            self.upscaler_ctx
        }
    }

    unsafe fn diffusion_ctx(&mut self, vae_decode_only: bool) -> *mut sd_ctx_t {
        if self.diffusion_ctx.is_none() {
            let ctx = new_sd_ctx(
                self.model.as_ptr(),
                self.clip_l.as_ptr(),
                self.clip_g.as_ptr(),
                self.t5xxl.as_ptr(),
                self.diffusion_model.as_ptr(),
                self.vae.as_ptr(),
                self.taesd.as_ptr(),
                self.control_net.as_ptr(),
                self.lora_model.as_ptr(),
                self.embeddings.as_ptr(),
                self.stacked_id_embd.as_ptr(),
                vae_decode_only,
                self.vae_tiling,
                false,
                self.n_threads,
                self.weight_type,
                self.rng,
                self.schedule,
                self.clip_on_cpu,
                self.control_net_cpu,
                self.vae_on_cpu,
                self.flash_attention,
            );
            self.diffusion_ctx = Some(ctx)
        }
        self.diffusion_ctx.unwrap()
    }
}

impl Drop for ModelConfig {
    fn drop(&mut self) {
        //Clean-up CTX section
        unsafe {
            if let Some(sd_ctx) = self.diffusion_ctx {
                free_sd_ctx(sd_ctx);
            }

            if let Some(upscaler_ctx) = self.upscaler_ctx {
                free_upscaler_ctx(upscaler_ctx);
            }
        }
    }
}

#[derive(Builder, Debug, Clone)]
#[builder(setter(into, strip_option), build_fn(validate = "Self::validate"))]
/// Config struct common to all diffusion methods
pub struct Config {
    /// Path to PHOTOMAKER input id images dir
    #[builder(default = "Default::default()")]
    input_id_images: CLibPath,

    /// Normalize PHOTOMAKER input id images
    #[builder(default = "false")]
    normalize_input: bool,

    /// Path to the input image, required by img2img
    #[builder(default = "Default::default()")]
    init_img: CLibPath,

    /// Path to image condition, control net
    #[builder(default = "Default::default()")]
    control_image: CLibPath,

    /// Path to write result image to (default: ./output.png)
    #[builder(default = "PathBuf::from(\"./output.png\")")]
    output: PathBuf,

    /// The prompt to render
    prompt: String,

    /// The negative prompt (default: "")
    #[builder(default = "\"\".into()")]
    negative_prompt: CLibString,

    /// Unconditional guidance scale (default: 7.0)
    #[builder(default = "7.0")]
    cfg_scale: f32,

    /// Guidance (default: 3.5)
    #[builder(default = "3.5")]
    guidance: f32,

    /// Strength for noising/unnoising (default: 0.75)
    #[builder(default = "0.75")]
    strength: f32,

    /// Strength for keeping input identity (default: 20%)
    #[builder(default = "20.0")]
    style_ratio: f32,

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

    /// Sampling-method (default: EULER_A)
    #[builder(default = "SampleMethod::EULER_A")]
    sampling_method: SampleMethod,

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
                "When batch_count > 1, ouput should point to folder and viceversa".to_owned(),
            ))
        }
    }
}

impl From<Config> for ConfigBuilder {
    fn from(value: Config) -> Self {
        let mut builder = ConfigBuilder::default();
        builder
            .input_id_images(value.input_id_images)
            .normalize_input(value.normalize_input)
            .init_img(value.init_img)
            .control_image(value.control_image)
            .output(value.output)
            .prompt(value.prompt)
            .negative_prompt(value.negative_prompt)
            .cfg_scale(value.cfg_scale)
            .strength(value.strength)
            .style_ratio(value.style_ratio)
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
            .canny(value.canny);

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

fn output_files(path: &Path, batch_size: i32) -> Vec<CLibPath> {
    if batch_size == 1 {
        vec![path.into()]
    } else {
        (1..=batch_size)
            .map(|id| path.join(format!("output_{id}.png")).into())
            .collect()
    }
}

unsafe fn upscale(
    upscale_repeats: i32,
    upscaler_ctx: Option<*mut upscaler_ctx_t>,
    data: sd_image_t,
) -> Result<sd_image_t, DiffusionError> {
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

/// Generate an image with a prompt
pub fn txt2img(config: &mut Config, model_config: &mut ModelConfig) -> Result<(), DiffusionError> {
    let prompt: CLibString = match &model_config.prompt_suffix {
        Some(suffix) => format!("{} {suffix}", &config.prompt),
        None => config.prompt.clone(),
    }
    .into();
    let files = output_files(&config.output, config.batch_count);
    unsafe {
        let sd_ctx = model_config.diffusion_ctx(true);
        let upscaler_ctx = model_config.upscaler_ctx();

        let slice = diffusion_rs_sys::txt2img(
            sd_ctx,
            prompt.as_ptr(),
            config.negative_prompt.as_ptr(),
            config.clip_skip as i32,
            config.cfg_scale,
            config.guidance,
            config.width,
            config.height,
            config.sampling_method,
            config.steps,
            config.seed,
            config.batch_count,
            null(),
            config.control_strength,
            config.style_ratio,
            config.normalize_input,
            config.input_id_images.as_ptr(),
            config.skip_layer.as_mut_ptr(),
            config.skip_layer.len(),
            config.slg_scale,
            config.skip_layer_start,
            config.skip_layer_end,
        );
        if slice.is_null() {
            return Err(DiffusionError::Forward);
        }
        for (id, (img, path)) in slice::from_raw_parts(slice, config.batch_count as usize)
            .iter()
            .zip(files)
            .enumerate()
        {
            match upscale(model_config.upscale_repeats, upscaler_ctx, *img) {
                Ok(img) => {
                    let status = stbi_write_png_custom(
                        path.as_ptr(),
                        img.width as i32,
                        img.height as i32,
                        img.channel as i32,
                        img.data as *const c_void,
                        0,
                    );
                    if status == 0 {
                        return Err(DiffusionError::StoreImages(id, config.batch_count));
                    }
                }
                Err(err) => {
                    return Err(err);
                }
            }
        }

        //Clean-up slice section
        free(slice as *mut c_void);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::{
        api::{ConfigBuilderError, ModelConfigBuilder},
        util::download_file_hf_hub,
    };

    use super::{txt2img, ConfigBuilder};

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
    fn test_txt2img() {
        let model_path =
            download_file_hf_hub("CompVis/stable-diffusion-v-1-4-original", "sd-v1-4.ckpt")
                .unwrap();

        let upscaler_path = download_file_hf_hub(
            "ximso/RealESRGAN_x4plus_anime_6B",
            "RealESRGAN_x4plus_anime_6B.pth",
        )
        .unwrap();
        let mut config = ConfigBuilder::default()
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

        txt2img(&mut config, &mut model_config).unwrap();
        let mut config2 = ConfigBuilder::from(config)
            .prompt("a lovely duck drinking water from a straw")
            .output(PathBuf::from("./output_2.png"))
            .build()
            .unwrap();
        txt2img(&mut config2, &mut model_config).unwrap();
    }
}
