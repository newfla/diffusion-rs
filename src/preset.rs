use derive_builder::Builder;
use hf_hub::api::sync::ApiError;
use subenum::subenum;

use crate::{
    api::{Config, ConfigBuilder, ConfigBuilderError, ModelConfig, ModelConfigBuilder},
    preset_builder::{
        chroma, chroma_radiance, diff_instruct_star, flux_1_dev, flux_1_mini, flux_1_schnell,
        flux_2_dev, juggernaut_xl_11, nitro_sd_realism, nitro_sd_vibrant, sd_turbo, sdxl_base_1_0,
        sdxl_turbo_1_0_fp16, ssd_1b, stable_diffusion_1_4, stable_diffusion_1_5,
        stable_diffusion_2_1, stable_diffusion_3_5_large_fp16,
        stable_diffusion_3_5_large_turbo_fp16, stable_diffusion_3_5_medium_fp16,
        stable_diffusion_3_medium_fp16, z_image_turbo,
    },
};

#[non_exhaustive]
#[allow(non_camel_case_types)]
#[subenum(
    Flux1Weight,
    Flux1MiniWeight,
    ChromaWeight,
    NitroSDRealismWeight,
    NitroSDVibrantWeight,
    DiffInstructStarWeight,
    ChromaRadianceWeight,
    SSD1BWeight,
    Flux2Weight,
    ZImageTurboWeight
)]
#[derive(Debug, Clone, Copy)]
/// Model weight types
pub enum WeightType {
    #[subenum(Flux1MiniWeight)]
    F32,
    #[subenum(
        NitroSDRealismWeight,
        NitroSDVibrantWeight,
        DiffInstructStarWeight,
        SSD1BWeight
    )]
    F16,
    #[subenum(
        Flux1Weight,
        ChromaWeight,
        NitroSDRealismWeight,
        NitroSDVibrantWeight,
        DiffInstructStarWeight,
        Flux2Weight,
        ZImageTurboWeight
    )]
    Q4_0,
    #[subenum(Flux2Weight)]
    Q4_1,
    #[subenum(
        NitroSDRealismWeight,
        NitroSDVibrantWeight,
        DiffInstructStarWeight,
        Flux2Weight,
        ZImageTurboWeight
    )]
    Q5_0,
    #[subenum(Flux2Weight)]
    Q5_1,
    #[subenum(
        Flux1Weight,
        Flux1MiniWeight,
        ChromaWeight,
        NitroSDRealismWeight,
        NitroSDVibrantWeight,
        DiffInstructStarWeight,
        ChromaRadianceWeight,
        Flux2Weight,
        ZImageTurboWeight
    )]
    Q8_0,
    Q8_1,
    #[subenum(
        Flux1Weight,
        Flux1MiniWeight,
        NitroSDRealismWeight,
        NitroSDVibrantWeight,
        DiffInstructStarWeight,
        Flux2Weight,
        ZImageTurboWeight
    )]
    Q2_K,
    #[subenum(
        Flux1Weight,
        Flux1MiniWeight,
        NitroSDRealismWeight,
        NitroSDVibrantWeight,
        DiffInstructStarWeight,
        ZImageTurboWeight,
        Flux2Weight
    )]
    Q3_K,
    #[subenum(Flux1Weight, ZImageTurboWeight, Flux2Weight)]
    Q4_K,
    #[subenum(Flux1MiniWeight, Flux2Weight)]
    Q5_K,
    #[subenum(
        Flux1MiniWeight,
        NitroSDRealismWeight,
        NitroSDVibrantWeight,
        DiffInstructStarWeight,
        Flux2Weight,
        ZImageTurboWeight
    )]
    Q6_K,
    Q8_K,
    IQ2_XXS,
    IQ2_XS,
    IQ3_XXS,
    IQ1_S,
    IQ4_NL,
    IQ3_S,
    IQ2_S,
    IQ4_XS,
    I8,
    I16,
    I32,
    I64,
    F64,
    IQ1_M,
    #[subenum(
        Flux1MiniWeight,
        ChromaWeight,
        ChromaRadianceWeight,
        Flux2Weight,
        ZImageTurboWeight
    )]
    BF16,
    TQ1_0,
    TQ2_0,
    MXFP4,
    #[subenum(SSD1BWeight)]
    F8_E4M3,
}

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
    /// Vae-tiling enabled. 1024x1024. Enabled [crate::api::SampleMethod::EULER_SAMPLE_METHOD]. 30 steps.
    StableDiffusion3MediumFp16,
    /// Requires access rights to <https://huggingface.co/stabilityai/stable-diffusion-3.5-medium> providing a token via [crate::util::set_hf_token]
    /// Vae-tiling enabled. 1024x1024. Enabled [crate::api::SampleMethod::EULER_SAMPLE_METHOD]. cfg_scale 4.5. 40 steps.
    StableDiffusion3_5MediumFp16,
    /// Requires access rights to <https://huggingface.co/stabilityai/stable-diffusion-3.5-large> providing a token via [crate::util::set_hf_token]
    /// Vae-tiling enabled. 1024x1024. Enabled [crate::api::SampleMethod::EULER_SAMPLE_METHOD]. cfg_scale 4.5. 28 steps.
    StableDiffusion3_5LargeFp16,
    /// Requires access rights to <https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo> providing a token via [crate::util::set_hf_token]
    /// Vae-tiling enabled. 1024x1024. Enabled [crate::api::SampleMethod::EULER_SAMPLE_METHOD]. cfg_scale 0. 4 steps.
    StableDiffusion3_5LargeTurboFp16,
    SDXLBase1_0,
    /// cfg_scale 1. guidance 0. 4 steps
    SDTurbo,
    /// cfg_scale 1. guidance 0. 4 steps
    SDXLTurbo1_0Fp16,
    /// Requires access rights to <https://huggingface.co/black-forest-labs/FLUX.1-dev> providing a token via [crate::util::set_hf_token]
    /// Vae-tiling enabled. 1024x1024. Enabled [crate::api::SampleMethod::EULER_SAMPLE_METHOD]. 28 steps.
    Flux1Dev(Flux1Weight),
    /// Requires access rights to <https://huggingface.co/black-forest-labs/FLUX.1-schnell> providing a token via [crate::util::set_hf_token]
    /// Vae-tiling enabled. 1024x1024. Enabled [crate::api::SampleMethod::EULER_SAMPLE_METHOD]. 4 steps.
    Flux1Schnell(Flux1Weight),
    /// A 3.2B param rectified flow transformer distilled from FLUX.1-dev <https://huggingface.co/TencentARC/flux-mini> <https://huggingface.co/HyperX-Sentience/Flux-Mini-GGUF>
    /// Vae-tiling enabled. 512x512. Enabled [crate::api::SampleMethod::EULER_SAMPLE_METHOD]. cfg_scale 1. 20 steps.
    Flux1Mini(Flux1MiniWeight),
    /// Requires access rights to <https://huggingface.co/RunDiffusion/Juggernaut-XI-v11> providing a token via [crate::util::set_hf_token]
    /// Vae-tiling enabled. 1024x1024. Enabled [crate::api::SampleMethod::DPM2_SAMPLE_METHOD]. guidance 6. 20 steps
    JuggernautXL11,
    /// Chroma is an 8.9B parameter model based on FLUX.1-schnell
    /// Requires access rights to <https://huggingface.co/black-forest-labs/FLUX.1-dev> providing a token via [crate::util::set_hf_token]
    /// Vae-tiling enabled. 512x512. Enabled [crate::api::SampleMethod::EULER_SAMPLE_METHOD]. cfg_scale 4. 20 steps
    Chroma(ChromaWeight),
    /// sgm_uniform scheduler. cfg_scale 1. timestep_shift 250. 1 steps. 1024x1024
    NitroSDRealism(NitroSDRealismWeight),
    /// sgm_uniform scheduler. cfg_scale 1. timestep_shift 500. 1 steps. 1024x1024
    NitroSDVibrant(NitroSDVibrantWeight),
    /// sgm_uniform scheduler. cfg_scale 1. timestep_shift 400. 1 steps. 1024x1024
    DiffInstructStar(DiffInstructStarWeight),
    /// Enabled [crate::api::SampleMethod::EULER_SAMPLE_METHOD]. cfg_scale 4.0. 20 steps.
    ChromaRadiance(ChromaRadianceWeight),
    /// cfg_scale 9.0. 20 steps. 1024x1024
    SSD1B(SSD1BWeight),
    /// Requires access rights to <https://huggingface.co/black-forest-labs/FLUX.2-dev> providing a token via [crate::util::set_hf_token]
    /// Enabled [crate::api::SampleMethod::EULER_SAMPLE_METHOD]. cfg_scale 1.0. Flash attention enabled. Offload params to CPU enabled. 20 steps. 512x512. Vae-tiling enabled.
    Flux2Dev(Flux2Weight),
    /// Requires access rights to <https://huggingface.co/black-forest-labs/FLUX.1-schnell> providing a token via [crate::util::set_hf_token]
    /// cfg_scale 1.0. 9 steps. Flash attention enabled. 1024x1024. Vae-tiling enabled.
    ZImageTurbo(ZImageTurboWeight),
}

impl Preset {
    fn try_configs_builder(self) -> Result<(ConfigBuilder, ModelConfigBuilder), ApiError> {
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
            Preset::JuggernautXL11 => juggernaut_xl_11(),
            Preset::Flux1Mini(sd_type_t) => flux_1_mini(sd_type_t),
            Preset::Chroma(sd_type_t) => chroma(sd_type_t),
            Preset::NitroSDRealism(sd_type_t) => nitro_sd_realism(sd_type_t),
            Preset::NitroSDVibrant(sd_type_t) => nitro_sd_vibrant(sd_type_t),
            Preset::DiffInstructStar(sd_type_t) => diff_instruct_star(sd_type_t),
            Preset::ChromaRadiance(sd_type_t) => chroma_radiance(sd_type_t),
            Preset::SSD1B(sd_type_t) => ssd_1b(sd_type_t),
            Preset::Flux2Dev(sd_type_t) => flux_2_dev(sd_type_t),
            Preset::ZImageTurbo(sd_type_t) => z_image_turbo(sd_type_t),
        }
    }
}

/// Configs tuple used by [crate::modifier]
pub type ConfigsBuilder = (ConfigBuilder, ModelConfigBuilder);

/// Returned by [PresetBuilder::build]
pub type Configs = (Config, ModelConfig);

/// Helper functions that modifies the [ConfigBuilder] See [crate::modifier]
pub type Modifier = fn(ConfigsBuilder) -> Result<ConfigsBuilder, ApiError>;

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
    modifiers: Vec<Modifier>,
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

    pub fn build(&mut self) -> Result<Configs, ConfigBuilderError> {
        let preset = self.internal_build()?;
        let configs: ConfigsBuilder = preset
            .try_into()
            .map_err(|err: ApiError| ConfigBuilderError::ValidationError(err.to_string()))?;
        let config = configs.0.build()?;
        let config_model = configs.1.build()?;

        Ok((config, config_model))
    }
}

impl TryFrom<PresetConfig> for ConfigsBuilder {
    type Error = ApiError;

    fn try_from(value: PresetConfig) -> Result<Self, Self::Error> {
        let mut configs_builder = value.preset.try_configs_builder()?;
        for m in value.modifiers {
            configs_builder = m(configs_builder)?;
        }
        configs_builder.0.prompt(value.prompt);
        Ok(configs_builder)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        api::gen_img,
        preset::{
            ChromaRadianceWeight, ChromaWeight, DiffInstructStarWeight, Flux1MiniWeight,
            Flux1Weight, Flux2Weight, NitroSDRealismWeight, NitroSDVibrantWeight, SSD1BWeight,
            ZImageTurboWeight,
        },
        util::set_hf_token,
    };

    use super::{Preset, PresetBuilder};
    static PROMPT: &str = "a lovely dynosaur made by crochet";

    fn run(preset: Preset) {
        let (config, mut model_config) = PresetBuilder::default()
            .preset(preset)
            .prompt(PROMPT)
            .build()
            .unwrap();
        gen_img(&config, &mut model_config).unwrap();
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
        run(Preset::Flux1Dev(Flux1Weight::Q2_K));
    }

    #[ignore]
    #[test]
    fn test_flux_1_schnell() {
        set_hf_token(include_str!("../token.txt"));
        run(Preset::Flux1Schnell(Flux1Weight::Q2_K));
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

    #[ignore]
    #[test]
    fn test_juggernaut_xl_11() {
        set_hf_token(include_str!("../token.txt"));
        run(Preset::JuggernautXL11);
    }

    #[ignore]
    #[test]
    fn test_flux_1_mini() {
        set_hf_token(include_str!("../token.txt"));
        run(Preset::Flux1Mini(Flux1MiniWeight::Q2_K));
    }

    #[ignore]
    #[test]
    fn test_chroma() {
        set_hf_token(include_str!("../token.txt"));
        run(Preset::Chroma(ChromaWeight::Q4_0));
    }

    #[ignore]
    #[test]
    fn test_nitro_sd_realism() {
        run(Preset::NitroSDRealism(NitroSDRealismWeight::Q8_0));
    }

    #[ignore]
    #[test]
    fn test_nitro_sd_vibrant() {
        run(Preset::NitroSDVibrant(NitroSDVibrantWeight::Q8_0));
    }

    #[ignore]
    #[test]
    fn test_diff_instruct_star() {
        run(Preset::DiffInstructStar(DiffInstructStarWeight::Q8_0));
    }

    #[ignore]
    #[test]
    fn test_chroma_radiance() {
        run(Preset::ChromaRadiance(ChromaRadianceWeight::Q8_0));
    }

    #[ignore]
    #[test]
    fn test_ssd_1b() {
        run(Preset::SSD1B(SSD1BWeight::F8_E4M3));
    }

    #[ignore]
    #[test]
    fn test_flux_2_dev() {
        set_hf_token(include_str!("../token.txt"));
        run(Preset::Flux2Dev(Flux2Weight::Q2_K));
    }

    #[ignore]
    #[test]
    fn test_z_image_turbo() {
        set_hf_token(include_str!("../token.txt"));
        run(Preset::ZImageTurbo(ZImageTurboWeight::Q2_K));
    }
}
