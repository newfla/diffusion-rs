use hf_hub::api::sync::ApiError;

use crate::{
    api::{self, ConfigBuilder},
    preset_builder::{
        flux_1_dev, flux_1_schnell, sd_turbo, sdxl_base_1_0, sdxl_turbo_1_0_fp16,
        stable_diffusion_1_4, stable_diffusion_1_5, stable_diffusion_2_1,
        stable_diffusion_3_5_large_fp16, stable_diffusion_3_5_large_turbo_fp16,
        stable_diffusion_3_5_medium_fp16, stable_diffusion_3_medium_fp16,
    },
};

#[non_exhaustive]
#[derive(Debug)]
/// Models ready to use
pub enum Preset {
    StableDiffusion1_4,
    StableDiffusion1_5,
    /// <https://huggingface.co/stabilityai/stable-diffusion-2-1> model.
    ///  Vae-tiling enabled. 768x768.
    StableDiffusion2_1,
    /// Requires access right to <https://huggingface.co/stabilityai/stable-diffusion-3-medium> providing a token via [crate::util::set_hf_token]
    /// Vae-tiling enabled. 1024x1024. Enabled [crate::api::SampleMethod::EULER]. 30 steps.
    StableDiffusion3MediumFp16,
    /// Requires access right to <https://huggingface.co/stabilityai/stable-diffusion-3.5-medium> providing a token via [crate::util::set_hf_token]
    /// Vae-tiling enabled. 1024x1024. Enabled [crate::api::SampleMethod::EULER]. cfg_scale 4.5. 28 steps.
    StableDiffusion3_5MediumFp16,
    /// Requires access right to <https://huggingface.co/stabilityai/stable-diffusion-3.5-large> providing a token via [crate::util::set_hf_token]
    /// Vae-tiling enabled. 1024x1024. Enabled [crate::api::SampleMethod::EULER]. cfg_scale 4.5. 28 steps.
    StableDiffusion3_5LargeFp16,
    /// Requires access right to <https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo> providing a token via [crate::util::set_hf_token]
    /// Vae-tiling enabled. 1024x1024. Enabled [crate::api::SampleMethod::EULER].  4 steps.
    StableDiffusion3_5LargeTurboFp16,
    SDXLBase1_0,
    /// cfg_scale 1. guidance 0. 4 Steps
    SDTurbo,
    /// cfg_scale 1. guidance 0. 4 Steps
    SDXLTurbo1_0Fp16,
    ///  Requires access right to <https://huggingface.co/black-forest-labs/FLUX.1-dev> providing a token via [crate::util::set_hf_token]
    /// Vae-tiling enabled. 1024x1024. Enabled [crate::api::SampleMethod::EULER]. 28 steps.
    Flux1Dev(api::WeightType),
    ///  Requires access right to <https://huggingface.co/black-forest-labs/FLUX.1-schnell> providing a token via [crate::util::set_hf_token]
    /// Vae-tiling enabled. 1024x1024. Enabled [crate::api::SampleMethod::EULER]. 4 steps.
    Flux1Schnell(api::WeightType),
}

impl Preset {
    pub fn build(self, prompt: &str) -> ConfigBox {
        let config_builder: Box<dyn FnOnce() -> Result<ConfigBuilder, ApiError>> = match self {
            Preset::StableDiffusion1_4 => Box::new(stable_diffusion_1_4),
            Preset::StableDiffusion1_5 => Box::new(stable_diffusion_1_5),
            Preset::StableDiffusion2_1 => Box::new(stable_diffusion_2_1),
            Preset::StableDiffusion3MediumFp16 => Box::new(stable_diffusion_3_medium_fp16),
            Preset::SDXLBase1_0 => Box::new(sdxl_base_1_0),
            Preset::Flux1Dev(sd_type_t) => Box::new(move || flux_1_dev(sd_type_t)),
            Preset::Flux1Schnell(sd_type_t) => Box::new(move || flux_1_schnell(sd_type_t)),
            Preset::SDTurbo => Box::new(sd_turbo),
            Preset::SDXLTurbo1_0Fp16 => Box::new(sdxl_turbo_1_0_fp16),
            Preset::StableDiffusion3_5LargeFp16 => Box::new(stable_diffusion_3_5_large_fp16),
            Preset::StableDiffusion3_5MediumFp16 => Box::new(stable_diffusion_3_5_medium_fp16),
            Preset::StableDiffusion3_5LargeTurboFp16 => {
                Box::new(stable_diffusion_3_5_large_turbo_fp16)
            }
        };
        ConfigBox {
            prompt: prompt.to_owned(),
            config_builder: config_builder,
            modifiers: Vec::new(),
        }
    }
}

/// Helper functions that modifies the [crate::api::ConfigBuilder] See [crate::modifier]
pub type Modifier = fn(ConfigBuilder) -> Result<ConfigBuilder, ApiError>;

/// Helper struct for [crate::api::ConfigBuilder]
pub struct ConfigBox {
    pub(crate) prompt: String,
    pub(crate) config_builder: Box<dyn FnOnce() -> Result<ConfigBuilder, ApiError>>,
    pub(crate) modifiers: Vec<fn(ConfigBuilder) -> Result<ConfigBuilder, ApiError>>,
}

impl ConfigBox {
    /// Add modifier that will applied in sequence
    pub fn with_modifier(&mut self, f: Modifier) {
        self.modifiers.push(f);
    }
}

impl TryFrom<ConfigBox> for ConfigBuilder {
    type Error = ApiError;

    fn try_from(value: ConfigBox) -> Result<Self, Self::Error> {
        let mut config_builder = (value.config_builder)()?;
        for m in value.modifiers {
            config_builder = m(config_builder)?;
        }
        config_builder.prompt(value.prompt);
        Ok(config_builder)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        api::{self, txt2img, ConfigBuilder},
        util::set_hf_token,
    };

    use super::Preset;
    static PROMPT: &str = "a lovely duck drinking water from a bottle";

    fn run(preset: Preset) {
        let config_builder: ConfigBuilder = preset.build(PROMPT).try_into().unwrap();

        let config = config_builder.build().unwrap();
        txt2img(config).unwrap();
    }

    #[ignore]
    #[test]
    fn test_stable_diffusion_1_4() {
        run(Preset::StableDiffusion1_4);
    }

    #[ignore]
    #[test]
    fn test_stable_diffusion_1_5() {
        run(Preset::StableDiffusion1_5);
    }

    #[ignore]
    #[test]
    fn test_stable_diffusion_2_1() {
        run(Preset::StableDiffusion2_1);
    }

    #[ignore]
    #[test]
    fn test_stable_diffusion_3_medium_fp16() {
        set_hf_token(include_str!("../token.txt"));
        run(Preset::StableDiffusion3MediumFp16);
    }

    #[ignore]
    #[test]
    fn test_sdxl_base_1_0() {
        run(Preset::SDXLBase1_0);
    }

    #[ignore]
    #[test]
    fn test_flux_1_dev() {
        set_hf_token(include_str!("../token.txt"));
        run(Preset::Flux1Dev(api::WeightType::SD_TYPE_Q2_K));
    }

    #[ignore]
    #[test]
    fn test_flux_1_schnell() {
        set_hf_token(include_str!("../token.txt"));
        run(Preset::Flux1Schnell(api::WeightType::SD_TYPE_Q2_K));
    }

    #[ignore]
    #[test]
    fn test_sd_turbo() {
        run(Preset::SDTurbo);
    }

    #[ignore]
    #[test]
    fn test_sdxl_turbo_1_0_fp16() {
        run(Preset::SDXLTurbo1_0Fp16);
    }

    #[ignore]
    #[test]
    fn test_stable_diffusion_3_5_medium_fp16() {
        set_hf_token(include_str!("../token.txt"));
        run(Preset::StableDiffusion3_5MediumFp16);
    }

    #[ignore]
    #[test]
    fn test_stable_diffusion_3_5_large_fp16() {
        set_hf_token(include_str!("../token.txt"));
        run(Preset::StableDiffusion3_5LargeFp16);
    }

    #[ignore]
    #[test]
    fn test_stable_diffusion_3_5_large_turbo_fp16() {
        set_hf_token(include_str!("../token.txt"));
        run(Preset::StableDiffusion3_5LargeTurboFp16);
    }
}
