use std::ffi::c_char;
use std::ffi::c_void;
use std::ffi::CString;
use std::io::Cursor;
use std::path::Path;
use std::path::PathBuf;
use std::ptr::null;
use std::slice;

use derive_builder::Builder;

use diffusion_rs_sys::free_sd_ctx;
use diffusion_rs_sys::get_num_physical_cores;

use diffusion_rs_sys::new_sd_ctx;
/// Specify the range function
pub use diffusion_rs_sys::rng_type_t as RngFunction;

/// Sampling methods
pub use diffusion_rs_sys::sample_method_t as SampleMethod;

/// Denoiser sigma schedule
pub use diffusion_rs_sys::schedule_t as Schedule;

use diffusion_rs_sys::sd_ctx_t;
/// Weight type
pub use diffusion_rs_sys::sd_type_t as WeightType;
use diffusion_rs_sys::stbi_write_png_custom;

#[repr(i32)]
#[non_exhaustive]
#[derive(Debug, Default, Copy, Clone, Hash, PartialEq, Eq)]
/// Ignore last layers of CLIP network
pub enum ClipSkip {
    /// Will be [clip_skip_t::None] for SD1.x, [clip_skip_t::OneLayer] for SD2.x
    #[default]
    Unspecified = 0,
    None = 1,
    OneLayer = 2,
}

#[derive(Builder, Debug, Clone)]
#[builder(setter(into, strip_option), build_fn(validate = "Self::validate"))]
/// Config struct common to all diffusion methods
pub struct Config {
    /// Number of threads to use during computation (default: 0).
    /// If n_ threads <= 0, then threads will be set to the number of CPU physical cores.
    #[builder(default = "0", setter(custom))]
    n_threads: i32,

    /// Path to full model
    #[builder(default = "Default::default()")]
    model: CLibPath,

    /// Path to the standalone diffusion model
    #[builder(default = "Default::default()")]
    diffusion_model: CLibPath,

    /// path to the clip-l text encoder
    #[builder(default = "Default::default()")]
    clip_l: CLibPath,

    /// Path to the the t5xxl text encoder
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

    /// Path to PHOTOMAKER input id images dir
    #[builder(default = "Default::default()")]
    input_id_images: CLibPath,

    /// Normalize PHOTOMAKER input id images
    #[builder(default = "false")]
    normalize_input: bool,

    /// Path to esrgan model. Upscale images after generate, just RealESRGAN_x4plus_anime_6B supported by now
    #[builder(default = "Default::default()")]
    upscale_model: CLibPath,

    /// Run the ESRGAN upscaler this many times (default 1)
    #[builder(default = "0")]
    upscale_repeats: usize,

    /// Weight type. If not specified, the default is the type of the weight file
    #[builder(default = "WeightType::SD_TYPE_COUNT")]
    weight_type: WeightType,

    /// Lora model directory
    #[builder(default = "Default::default()")]
    lora_model: CLibPath,

    /// Path to the input image, required by img2img
    #[builder(default = "Default::default()")]
    init_img: CLibPath,

    /// Path to image condition, control net
    #[builder(default = "Default::default()")]
    control_image: CLibPath,

    /// Path to write result image to (default: ./output.png)
    #[builder(default = "PathBuf::from(\"./output.png\").into()")]
    output: CLibPath,

    /// The prompt to render
    prompt: CLibString,

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

    /// RNG (default: CUDA)
    #[builder(default = "RngFunction::CUDA_RNG")]
    rng: RngFunction,

    /// RNG seed (default: 42, use random seed for < 0)
    #[builder(default = "42")]
    seed: i64,

    /// Number of images to generate (default: 1)
    #[builder(default = "1")]
    batch_count: i32,

    /// Denoiser sigma schedule (default: DEFAULT)
    #[builder(default = "Schedule::DEFAULT")]
    schedule: Schedule,

    /// ignore last layers of CLIP network; 1 ignores none, 2 ignores one layer (default: -1)
    /// <= 0 represents unspecified, will be 1 for SD1.x, 2 for SD2.x
    #[builder(default = "ClipSkip::Unspecified")]
    clip_skip: ClipSkip,

    /// Process vae in tiles to reduce memory usage (default: false)
    #[builder(default = "false")]
    vae_tiling: bool,

    /// Keep vae in cpu (for low vram) (default: false)
    #[builder(default = "false")]
    vae_on_cpu: bool,

    /// keep clip in cpu (for low vram) (default: false)
    #[builder(default = "false")]
    clip_on_cpu: bool,

    /// Keep controlnet in cpu (for low vram) (default: false)
    #[builder(default = "false")]
    control_net_cpu: bool,

    /// Apply canny preprocessor (edge detection) (default: false)
    #[builder(default = "false")]
    canny: bool,
}

impl ConfigBuilder {
    fn custom_setter(&mut self, value: i32) {
        unsafe {
            self.n_threads = if value <= 0 {
                Some(get_num_physical_cores())
            } else {
                Some(value)
            }
        }
    }

    fn validate(&self) -> Result<(), String> {
        self.model
            .as_ref()
            .or(self.diffusion_model.as_ref())
            .map(|_| ())
            .ok_or("Model OR DiffusionModel must be valorized".to_string())
    }
}

impl Config {
    fn build_ctx(&self, vae_decode_only: bool) -> *mut sd_ctx_t {
        unsafe {
            new_sd_ctx(
                self.model.as_ptr(),
                self.clip_l.as_ptr(),
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
                true,
                self.n_threads,
                self.weight_type,
                self.rng,
                self.schedule,
                self.clip_on_cpu,
                self.control_net_cpu,
                self.vae_on_cpu,
            )
        }
    }
}

#[derive(Debug, Clone, Default)]
struct CLibString {
    data: CString,
}

impl CLibString {
    fn as_ptr(&self) -> *const c_char {
        self.data.as_ptr()
    }
}

impl From<&str> for CLibString {
    fn from(value: &str) -> Self {
        Self {
            data: CString::new(value).unwrap(),
        }
    }
}

#[derive(Debug, Clone, Default)]
struct CLibPath {
    path: CString,
}

impl CLibPath {
    fn as_ptr(&self) -> *const c_char {
        self.path.as_ptr()
    }
}

impl From<PathBuf> for CLibPath {
    fn from(value: PathBuf) -> Self {
        Self {
            path: CString::new(value.to_str().unwrap_or_default()).unwrap(),
        }
    }
}

impl From<&Path> for CLibPath {
    fn from(value: &Path) -> Self {
        Self {
            path: CString::new(value.to_str().unwrap_or_default()).unwrap(),
        }
    }
}

pub fn txt2img(config: Config) {
    let sd_ctx = config.build_ctx(true);
    unsafe {
        let slice = diffusion_rs_sys::txt2img(
            sd_ctx,
            config.prompt.as_ptr(),
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
        );
        let res = if slice.is_null() {
        } else {
            for img in slice::from_raw_parts(slice, config.batch_count as usize) {
                stbi_write_png_custom(
                    config.output.as_ptr(),
                    img.width as i32,
                    img.height as i32,
                    img.channel as i32,
                    img.data as *const c_void,
                    0,
                );
            }
        };
        free_sd_ctx(sd_ctx);
        res
    }
}

mod tests {
    use std::path::{Path, PathBuf};

    use hf_hub::api::sync::{Api, ApiError};

    use super::{txt2img, Config, ConfigBuilder};

    fn download_file_hf_hub(model: &str, file: &str) -> Result<PathBuf, ApiError> {
        let repo = Api::new()?.model(model.to_string());
        repo.get(file)
    }

    #[test]
    fn test_required_args_txt2img() {
        assert!(ConfigBuilder::default().build().is_err());
        assert!(ConfigBuilder::default()
            .model(Path::new("./test.ckpt"))
            .build()
            .is_err());

        assert!(ConfigBuilder::default()
            .prompt("a lovely cat driving a sport car")
            .build()
            .is_err());

        ConfigBuilder::default()
            .model(Path::new("./test.ckpt"))
            .prompt("a lovely cat driving a sport car")
            .build()
            .unwrap();
    }

    #[ignore]
    #[test]
    fn test_txt2img() {
        //let model_path = PathBuf::from("/home/flavio/stable-diffusion.cpp/sd-v1-4.ckpt");
        let model_path =
            download_file_hf_hub("CompVis/stable-diffusion-v-1-4-original", "sd-v1-4.ckpt")
                .unwrap();
        let config = ConfigBuilder::default()
            .model(model_path)
            //.prompt("a lovely cat driving a sport car")
            .prompt("a lovely duck drinking water from a bottle")
            .output(Path::new("/home/flavio/test1.png"))
            .build()
            .unwrap();
        txt2img(config);
    }
}
