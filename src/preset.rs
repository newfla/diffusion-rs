use derive_builder::Builder;
use hf_hub::api::sync::ApiError;

use crate::{
    api::{self, Config, ConfigBuilder, ConfigBuilderError},
    preset_builder::{
        flux_1_dev, flux_1_schnell, sd_turbo, sdxl_base_1_0, sdxl_turbo_1_0_fp16,
        stable_diffusion_1_4, stable_diffusion_1_5, stable_diffusion_2_1,
        stable_diffusion_3_5_large_fp16, stable_diffusion_3_5_large_turbo_fp16,
        stable_diffusion_3_5_medium_fp16, stable_diffusion_3_medium_fp16,
    },
};

#[non_exhaustive]
#[derive(Debug, Clone, Copy)]
/// Models ready to use
pub enum Preset {
    StableDiffusion1_4,
    StableDiffusion1_5,
    /// <https://huggingface.co/stabilityai/stable-diffusion-2-1> model.
    ///  Vae-tiling enabled. 768x768.
    StableDiffusion2_1,
    /// Requires access rights to <https://huggingface.co/stabilityai/stable-diffusion-3-medium> providing a token via [crate::util::set_hf_token]
    /// Vae-tiling enabled. 1024x1024. Enabled [api::SampleMethod::EULER]. 30 steps.
    StableDiffusion3MediumFp16,
    /// Requires access rights to <https://huggingface.co/stabilityai/stable-diffusion-3.5-medium> providing a token via [crate::util::set_hf_token]
    /// Vae-tiling enabled. 1024x1024. Enabled [api::SampleMethod::EULER]. cfg_scale 4.5. 40 steps.
    StableDiffusion3_5MediumFp16,
    /// Requires access rights to <https://huggingface.co/stabilityai/stable-diffusion-3.5-large> providing a token via [crate::util::set_hf_token]
    /// Vae-tiling enabled. 1024x1024. Enabled [api::SampleMethod::EULER]. cfg_scale 4.5. 28 steps.
    StableDiffusion3_5LargeFp16,
    /// Requires access rights to <https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo> providing a token via [crate::util::set_hf_token]
    /// Vae-tiling enabled. 1024x1024. Enabled [api::SampleMethod::EULER]. cfg_scale 0. 4 steps.
    StableDiffusion3_5LargeTurboFp16,
    SDXLBase1_0,
    /// cfg_scale 1. guidance 0. 4 Steps
    SDTurbo,
    /// cfg_scale 1. guidance 0. 4 Steps
    SDXLTurbo1_0Fp16,
    /// Requires access rights to <https://huggingface.co/black-forest-labs/FLUX.1-dev> providing a token via [crate::util::set_hf_token]
    /// Vae-tiling enabled. 1024x1024. Enabled [api::SampleMethod::EULER]. 28 steps.
    Flux1Dev(api::WeightType),
    /// Requires access rights to <https://huggingface.co/black-forest-labs/FLUX.1-schnell> providing a token via [crate::util::set_hf_token]
    /// Vae-tiling enabled. 1024x1024. Enabled [api::SampleMethod::EULER]. 4 steps.
    Flux1Schnell(api::WeightType),
}

impl Preset {
    fn try_config_builder(self) -> Result<ConfigBuilder, ApiError> {
        match self {
            Preset::StableDiffusion1_4 => stable_diffusion_1_4(),
            Preset::StableDiffusion1_5 => stable_diffusion_1_5(),
            Preset::StableDiffusion2_1 => stable_diffusion_2_1(),
            Preset::StableDiffusion3MediumFp16 => stable_diffusion_3_medium_fp16(),
            Preset::SDXLBase1_0 => sdxl_base_1_0(),
            Preset::Flux1Dev(sd_type_t) => flux_1_dev(sd_type_t),
            Preset::Flux1Schnell(sd_type_t) => flux_1_schnell(sd_type_t),
            Preset::SDTurbo => sd_turbo(),
            Preset::SDXLTurbo1_0Fp16 => sdxl_turbo_1_0_fp16(),
            Preset::StableDiffusion3_5LargeFp16 => stable_diffusion_3_5_large_fp16(),
            Preset::StableDiffusion3_5MediumFp16 => stable_diffusion_3_5_medium_fp16(),
            Preset::StableDiffusion3_5LargeTurboFp16 => stable_diffusion_3_5_large_turbo_fp16(),
        }
    }
}

/// Helper functions that modifies the [ConfigBuilder] See [crate::modifier]
pub type Modifier = fn(ConfigBuilder) -> Result<ConfigBuilder, ApiError>;

#[derive(Debug, Clone, Builder)]
#[builder(
    name = "PresetBuilder",
    setter(into),
    build_fn(name = "internal_build", private, error = "ConfigBuilderError")
)]
/// Helper struct for [ConfigBuilder]
pub struct PresetConfig {
    prompt: String,
    preset: Preset,
    #[builder(private, default = "Vec::new()")]
    modifiers: Vec<fn(ConfigBuilder) -> Result<ConfigBuilder, ApiError>>,
}

impl PresetBuilder {
    /// Add modifier that will apply in sequence
    pub fn with_modifier(&mut self, f: Modifier) -> &mut Self {
        if self.modifiers.is_none() {
            self.modifiers = Some(Vec::new());
        }
        self.modifiers.as_mut().unwrap().push(f);
        self
    }

    pub fn build(&mut self) -> Result<Config, ConfigBuilderError> {
        let preset = self.internal_build()?;
        let config: ConfigBuilder = preset
            .try_into()
            .map_err(|err: ApiError| ConfigBuilderError::ValidationError(err.to_string()))?;
        config.build()
    }
}

impl TryFrom<PresetConfig> for ConfigBuilder {
    type Error = ApiError;

    fn try_from(value: PresetConfig) -> Result<Self, Self::Error> {
        let mut config_builder = value.preset.try_config_builder()?;
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
        api::{self, txt2img},
        util::set_hf_token,
    };

    use super::{Preset, PresetBuilder};
    static PROMPT: &str = "a lovely duck drinking water from a bottle";

    fn run(preset: Preset) {
        let config = PresetBuilder::default()
            .preset(preset)
            .prompt(PROMPT)
            .build()
            .unwrap();
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
