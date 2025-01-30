use derive_builder::Builder;

use crate::utils::CLibPath;

/// Specify the range function
pub use diffusion_rs_sys::rng_type_t as RngFunction;

/// Denoiser sigma schedule
pub use diffusion_rs_sys::schedule_t as Schedule;

/// Weight type
pub use diffusion_rs_sys::sd_type_t as WeightType;

#[derive(Builder, Debug, Clone)]
#[builder(setter(into), build_fn(validate = "Self::validate"))]
/// Config struct common to all diffusion methods
pub struct ModelConfig {
    /// Path to full model
    #[builder(default = "Default::default()")]
    pub model: CLibPath,

    /// path to the clip-l text encoder
    #[builder(default = "Default::default()")]
    pub clip_l: CLibPath,

    /// path to the clip-g text encoder
    #[builder(default = "Default::default()")]
    pub clip_g: CLibPath,

    /// Path to the t5xxl text encoder
    #[builder(default = "Default::default()")]
    pub t5xxl: CLibPath,

    /// Path to the standalone diffusion model
    #[builder(default = "Default::default()")]
    pub diffusion_model: CLibPath,

    /// Path to vae
    #[builder(default = "Default::default()")]
    pub vae: CLibPath,

    /// Path to taesd. Using Tiny AutoEncoder for fast decoding (lower quality)
    #[builder(default = "Default::default()")]
    pub taesd: CLibPath,

    /// Path to control net model
    #[builder(default = "Default::default()")]
    pub control_net: CLibPath,

    /// Lora models directory
    #[builder(default = "Default::default()")]
    pub lora_model_dir: CLibPath,

    /// Path to embeddings directory
    #[builder(default = "Default::default()")]
    pub embeddings_dir: CLibPath,

    /// Path to PHOTOMAKER stacked id embeddings
    #[builder(default = "Default::default()")]
    pub stacked_id_embd_dir: CLibPath,

    //TODO: Add more info here for docs
    /// vae decode only (default: false)
    #[builder(default = "false")]
    pub vae_decode_only: bool,

    /// Process vae in tiles to reduce memory usage (default: false)
    #[builder(default = "false")]
    pub vae_tiling: bool,

    /// free memory of params immediately after forward (default: false)
    #[builder(default = "false")]
    pub free_params_immediately: bool,

    /// Number of threads to use during computation (default: 0).
    /// If n_ threads <= 0, then threads will be set to the number of CPU physical cores.
    #[builder(
        default = "std::thread::available_parallelism().map_or(1, |p| p.get() as i32)",
        setter(custom)
    )]
    pub n_threads: i32,

    /// Weight type. If not specified, the default is the type of the weight file
    #[builder(default = "WeightType::SD_TYPE_COUNT")]
    pub weight_type: WeightType,

    /// RNG type (default: CUDA)
    #[builder(default = "RngFunction::CUDA_RNG")]
    pub rng_type: RngFunction,

    /// Denoiser sigma schedule (default: DEFAULT)
    #[builder(default = "Schedule::DEFAULT")]
    pub schedule: Schedule,

    /// keep clip on cpu (for low vram) (default: false)
    #[builder(default = "false")]
    pub keep_clip_on_cpu: bool,

    /// Keep controlnet in cpu (for low vram) (default: false)
    #[builder(default = "false")]
    pub keep_control_net_cpu: bool,

    /// Keep vae on cpu (for low vram) (default: false)
    #[builder(default = "false")]
    pub keep_vae_on_cpu: bool,

    /// Use flash attention in the diffusion model (for low vram).
    /// Might lower quality, since it implies converting k and v to f16.
    /// This might crash if it is not supported by the backend.
    /// must have feature "flash_attention" enabled in the features.
    /// (default: false)
    #[builder(default = "false")]
    pub flash_attention: bool,
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
