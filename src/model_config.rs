use crate::types::{RngFunction, Schedule, WeightType};
use derive_builder::Builder;

use std::path::PathBuf;

#[derive(Builder, Clone, Debug)]
#[builder(setter(into), build_fn(validate = "Self::validate"))]
/// Config struct common to all diffusion methods
pub struct ModelConfig {
    /// Path to full model
    #[builder(default = "Default::default()")]
    pub model: PathBuf,

    /// path to the clip-l text encoder
    #[builder(default = "Default::default()")]
    pub clip_l: PathBuf,

    /// path to the clip-g text encoder
    #[builder(default = "Default::default()")]
    pub clip_g: PathBuf,

    /// Path to the t5xxl text encoder
    #[builder(default = "Default::default()")]
    pub t5xxl: PathBuf,

    /// Path to the standalone diffusion model
    #[builder(default = "Default::default()")]
    pub diffusion_model: PathBuf,

    /// Path to vae
    #[builder(default = "Default::default()")]
    pub vae: PathBuf,

    /// Path to taesd. Using Tiny AutoEncoder for fast decoding (lower quality)
    #[builder(default = "Default::default()")]
    pub taesd: PathBuf,

    /// Path to control net model
    #[builder(default = "Default::default()")]
    pub control_net: PathBuf,

    /// Lora models directory
    #[builder(default = "Default::default()")]
    pub lora_model_dir: PathBuf,

    /// Path to embeddings directory
    #[builder(default = "Default::default()")]
    pub embeddings_dir: PathBuf,

    /// Path to PHOTOMAKER stacked id embeddings
    #[builder(default = "Default::default()")]
    pub stacked_id_embd_dir: PathBuf,

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
    /// If n_threads <= 0, then threads will be set to the number of CPU physical cores.
    #[builder(
        default = "unsafe { diffusion_rs_sys::get_num_physical_cores() }",
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

unsafe impl Send for ModelConfig {}

impl ModelConfigBuilder {
    pub fn n_threads(&mut self, value: i32) -> &mut Self {
        self.n_threads = if value > 0 {
            Some(value)
        } else {
            Some(unsafe { diffusion_rs_sys::get_num_physical_cores() })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invalid_model_config() {
        let config = ModelConfigBuilder::default().build();
        assert!(config.is_err(), "ModelConfig should fail without a model");
    }

    #[test]
    fn test_valid_model_config() {
        let config = ModelConfigBuilder::default().model("./test.ckpt").build();
        assert!(config.is_ok(), "ModelConfig should succeed with model path");
    }
}
