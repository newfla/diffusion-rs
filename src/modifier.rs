use hf_hub::api::sync::ApiError;

use crate::{api::SampleMethod, preset::ConfigsBuilder, util::download_file_hf_hub};

/// Add the <https://huggingface.co/ximso/RealESRGAN_x4plus_anime_6B> upscaler
pub fn real_esrgan_x4plus_anime_6_b(
    mut builder: ConfigsBuilder,
) -> Result<ConfigsBuilder, ApiError> {
    let upscaler_path = download_file_hf_hub(
        "ximso/RealESRGAN_x4plus_anime_6B",
        "RealESRGAN_x4plus_anime_6B.pth",
    )?;
    builder.1.upscale_model(upscaler_path);
    Ok(builder)
}

/// Apply <https://huggingface.co/madebyollin/sdxl-vae-fp16-fix> to avoid black images with xl models
pub fn sdxl_vae_fp16_fix(mut builder: ConfigsBuilder) -> Result<ConfigsBuilder, ApiError> {
    let vae_path = download_file_hf_hub("madebyollin/sdxl-vae-fp16-fix", "sdxl.vae.safetensors")?;
    builder.1.vae(vae_path);
    Ok(builder)
}

/// Apply <https://huggingface.co/madebyollin/taesd> taesd autoencoder for faster decoding (SD v1/v2)
pub fn taesd(mut builder: ConfigsBuilder) -> Result<ConfigsBuilder, ApiError> {
    let taesd_path =
        download_file_hf_hub("madebyollin/taesd", "diffusion_pytorch_model.safetensors")?;
    builder.1.taesd(taesd_path);
    Ok(builder)
}

/// Apply <https://huggingface.co/madebyollin/taesdxl> taesd autoencoder for faster decoding (SDXL)
pub fn taesd_xl(mut builder: ConfigsBuilder) -> Result<ConfigsBuilder, ApiError> {
    let taesd_path =
        download_file_hf_hub("madebyollin/taesdxl", "diffusion_pytorch_model.safetensors")?;
    builder.1.taesd(taesd_path);
    Ok(builder)
}

/// Apply <https://huggingface.co/cqyan/hybrid-sd-tinyvae> taesd autoencoder for faster decoding (SD v1/v2)
pub fn hybrid_taesd(mut builder: ConfigsBuilder) -> Result<ConfigsBuilder, ApiError> {
    let taesd_path = download_file_hf_hub(
        "cqyan/hybrid-sd-tinyvae",
        "diffusion_pytorch_model.safetensors",
    )?;
    builder.1.taesd(taesd_path);
    Ok(builder)
}

/// Apply <https://huggingface.co/cqyan/hybrid-sd-tinyvae-xl> taesd autoencoder for faster decoding (SDXL)
pub fn hybrid_taesd_xl(mut builder: ConfigsBuilder) -> Result<ConfigsBuilder, ApiError> {
    let taesd_path = download_file_hf_hub(
        "cqyan/hybrid-sd-tinyvae-xl",
        "diffusion_pytorch_model.safetensors",
    )?;
    builder.1.taesd(taesd_path);
    Ok(builder)
}

/// Apply <https://huggingface.co/latent-consistency/lcm-lora-sdv1-5> to reduce inference steps for SD v1 between 2-8
/// cfg_scale 1. 4 steps.
pub fn lcm_lora_sd_1_5(mut builder: ConfigsBuilder) -> Result<ConfigsBuilder, ApiError> {
    let lora_path = download_file_hf_hub(
        "latent-consistency/lcm-lora-sdv1-5",
        "pytorch_lora_weights.safetensors",
    )?;
    builder.1.lora_model(&lora_path);
    builder.0.cfg_scale(1.).steps(4);
    Ok(builder)
}

/// Apply <https://huggingface.co/latent-consistency/lcm-lora-sdxl> to reduce inference steps for SD v1 between 2-8 (default 8)
/// Enabled [SampleMethod::LCM]. cfg_scale 2. 8 steps.
pub fn lcm_lora_sdxl_base_1_0(mut builder: ConfigsBuilder) -> Result<ConfigsBuilder, ApiError> {
    let lora_path = download_file_hf_hub(
        "latent-consistency/lcm-lora-sdxl",
        "pytorch_lora_weights.safetensors",
    )?;
    builder.1.lora_model(&lora_path);
    builder
        .0
        .cfg_scale(2.)
        .steps(8)
        .sampling_method(SampleMethod::LCM);
    Ok(builder)
}

/// Apply <https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/t5xxl_fp8_e4m3fn.safetensors> Fp8 t5xxl text encoder to reduce memory usage
pub fn t5xxl_fp8_flux_1(mut builder: ConfigsBuilder) -> Result<ConfigsBuilder, ApiError> {
    let t5xxl_path = download_file_hf_hub(
        "comfyanonymous/flux_text_encoders",
        "t5xxl_fp8_e4m3fn.safetensors",
    )?;

    builder.1.t5xxl(t5xxl_path);
    Ok(builder)
}

/// Apply <https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/t5xxl_fp16.safetensors>
/// Default for flux_1_dev/schnell
pub fn t5xxl_fp16_flux_1(mut builder: ConfigsBuilder) -> Result<ConfigsBuilder, ApiError> {
    let t5xxl_path = download_file_hf_hub(
        "comfyanonymous/flux_text_encoders",
        "t5xxl_fp16.safetensors",
    )?;

    builder.1.t5xxl(t5xxl_path);
    Ok(builder)
}

/// Apply <https://huggingface.co/Green-Sky/flux.1-schnell-GGUF/blob/main/t5xxl_q2_k.gguf>
pub fn t5xxl_q2_k_flux_1(mut builder: ConfigsBuilder) -> Result<ConfigsBuilder, ApiError> {
    let t5xxl_path = download_file_hf_hub("Green-Sky/flux.1-schnell-GGUF", "t5xxl_q2_k.gguf")?;

    builder.1.t5xxl(t5xxl_path);
    Ok(builder)
}

/// Apply <https://huggingface.co/Green-Sky/flux.1-schnell-GGUF/blob/main/t5xxl_q3_k.gguf>
pub fn t5xxl_q3_k_flux_1(mut builder: ConfigsBuilder) -> Result<ConfigsBuilder, ApiError> {
    let t5xxl_path = download_file_hf_hub("Green-Sky/flux.1-schnell-GGUF", "t5xxl_q3_k.gguf")?;

    builder.1.t5xxl(t5xxl_path);
    Ok(builder)
}

/// Apply <https://huggingface.co/Green-Sky/flux.1-schnell-GGUF/blob/main/t5xxl_q4_k.gguf>
/// Default for flux_1_mini
pub fn t5xxl_q4_k_flux_1(mut builder: ConfigsBuilder) -> Result<ConfigsBuilder, ApiError> {
    let t5xxl_path = download_file_hf_hub("Green-Sky/flux.1-schnell-GGUF", "t5xxl_q4_k.gguf")?;

    builder.1.t5xxl(t5xxl_path);
    Ok(builder)
}

/// Apply <https://huggingface.co/Green-Sky/flux.1-schnell-GGUF/blob/main/t5xxl_q8_0.gguf>
pub fn t5xxl_q8_0_flux_1(mut builder: ConfigsBuilder) -> Result<ConfigsBuilder, ApiError> {
    let t5xxl_path = download_file_hf_hub("Green-Sky/flux.1-schnell-GGUF", "t5xxl_q8_0.gguf")?;

    builder.1.t5xxl(t5xxl_path);
    Ok(builder)
}

pub fn offload_params_to_cpu(mut builder: ConfigsBuilder) -> Result<ConfigsBuilder, ApiError> {
    builder.1.offload_params_to_cpu(true);
    Ok(builder)
}

#[cfg(test)]
mod tests {
    use crate::{
        api::{self, gen_img},
        modifier::offload_params_to_cpu,
        preset::{Modifier, Preset, PresetBuilder},
        util::set_hf_token,
    };

    use super::{
        hybrid_taesd, hybrid_taesd_xl, lcm_lora_sd_1_5, lcm_lora_sdxl_base_1_0, taesd, taesd_xl,
    };

    static PROMPT: &str = "a lovely cat holding a sign says 'diffusion-rs'";

    fn run(preset: Preset, m: Modifier) {
        let (mut config, mut model_config) = PresetBuilder::default()
            .preset(preset)
            .prompt(PROMPT)
            .with_modifier(m)
            .build()
            .unwrap();
        gen_img(&mut config, &mut model_config).unwrap();
    }

    #[ignore]
    #[test]
    fn test_taesd() {
        run(Preset::StableDiffusion1_5, taesd);
    }

    #[ignore]
    #[test]
    fn test_taesd_xl() {
        run(Preset::SDXLTurbo1_0Fp16, taesd_xl);
    }

    #[ignore]
    #[test]
    fn test_hybrid_taesd() {
        run(Preset::StableDiffusion1_5, hybrid_taesd);
    }

    #[ignore]
    #[test]
    fn test_hybrid_taesd_xl() {
        run(Preset::SDXLTurbo1_0Fp16, hybrid_taesd_xl);
    }

    #[ignore]
    #[test]
    fn test_lcm_lora_sd_1_5() {
        run(Preset::StableDiffusion1_5, lcm_lora_sd_1_5);
    }

    #[ignore]
    #[test]
    fn test_lcm_lora_sdxl_base_1_0() {
        run(Preset::SDXLBase1_0, lcm_lora_sdxl_base_1_0);
    }

    #[ignore]
    #[test]
    fn test_offload_params_to_cpu() {
        set_hf_token(include_str!("../token.txt"));
        run(
            Preset::Flux1Schnell(api::WeightType::SD_TYPE_Q2_K),
            offload_params_to_cpu,
        );
    }
}
