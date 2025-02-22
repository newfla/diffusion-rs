use std::path::PathBuf;

use crate::utils::SampleMethod;
use derive_builder::Builder;
use image::RgbImage;

#[derive(Builder, Debug, Clone)]
#[builder(setter(into, strip_option), build_fn(validate = "Self::validate"))]
/// txt2img config
pub struct Txt2ImgConfig {
    /// Prompt to generate image from
    pub prompt: String,

    /// Suffix that needs to be added to prompt (e.g. lora model)
    #[builder(default = "Default::default()", private)]
    pub lora_prompt_suffix: Vec<String>,

    /// The negative prompt (default: "")
    #[builder(default = "\"\".into()")]
    pub negative_prompt: String,

    /// Ignore last layers of CLIP network; 1 ignores none, 2 ignores one layer (default: -1)
    /// <= 0 represents unspecified, will be 1 for SD1.x, 2 for SD2.x
    #[builder(default = "0")]
    pub clip_skip: i32,

    /// Unconditional guidance scale (default: 7.0)
    #[builder(default = "7.0")]
    pub cfg_scale: f32,

    /// Guidance (default: 3.5) for Flux/DiT models
    #[builder(default = "3.5")]
    pub guidance: f32,

    /// Image height, in pixel space (default: 512)
    #[builder(default = "512")]
    pub height: i32,

    /// Image width, in pixel space (default: 512)
    #[builder(default = "512")]
    pub width: i32,

    /// Sampling-method (default: EULER_A)
    #[builder(default = "SampleMethod::EULER_A")]
    pub sample_method: SampleMethod,

    /// Number of sample steps (default: 20)
    #[builder(default = "20")]
    pub sample_steps: i32,

    /// RNG seed (default: 42, use random seed for < 0)
    #[builder(default = "42")]
    pub seed: i64,

    /// Number of images to generate (default: 1)
    #[builder(default = "1")]
    pub batch_count: i32,

    #[builder(default = "None")]
    pub control_cond: Option<RgbImage>,

    /// Strength to apply Control Net (default: 0.9)
    /// 1.0 corresponds to full destruction of information in init
    #[builder(default = "0.9")]
    pub control_strength: f32,

    /// Strength for keeping input identity (default: 20%)
    #[builder(default = "20.0")]
    pub style_strength: f32,

    /// Normalize PHOTOMAKER input id images
    #[builder(default = "false")]
    pub normalize_input: bool,

    /// Path to PHOTOMAKER input id images dir
    #[builder(default = "Default::default()")]
    pub input_id_images: PathBuf,

    /// Layers to skip for SLG steps: (default: [7,8,9])
    #[builder(default = "vec![7, 8, 9]")]
    pub skip_layer: Vec<i32>,

    /// skip layer guidance (SLG) scale, only for DiT models: (default: 0)
    /// 0 means disabled, a value of 2.5 is nice for sd3.5 medium
    #[builder(default = "0.")]
    pub slg_scale: f32,

    /// SLG enabling point: (default: 0.01)
    #[builder(default = "0.01")]
    pub skip_layer_start: f32,

    /// SLG disabling point: (default: 0.2)
    #[builder(default = "0.2")]
    pub skip_layer_end: f32,
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

    pub fn add_lora_model(&mut self, filename: &str, strength: f32) -> &mut Self {
        self.lora_prompt_suffix
            .get_or_insert_with(Vec::new)
            .push(format!("<lora:{filename}:{strength}>"));
        self
    }
}
