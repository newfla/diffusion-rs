use hf_hub::api::sync::ApiError;

use crate::{
    api::{PreviewType, SampleMethod},
    preset::ConfigsBuilder,
    util::download_file_hf_hub,
};

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

/// Apply <https://huggingface.co/latent-consistency/lcm-lora-sdv1-5> to reduce inference steps for SD v1 between 2-8 (default 8)
/// cfg_scale 1. 8 steps.
pub fn lcm_lora_sd_1_5(mut builder: ConfigsBuilder) -> Result<ConfigsBuilder, ApiError> {
    let lora_path = download_file_hf_hub(
        "latent-consistency/lcm-lora-sdv1-5",
        "pytorch_lora_weights.safetensors",
    )?;
    builder.1.lora_model(&lora_path);
    builder.0.cfg_scale(1.).steps(8);
    Ok(builder)
}

/// Apply <https://huggingface.co/latent-consistency/lcm-lora-sdxl> to reduce inference steps for SD v1 between 2-8 (default 8)
/// Enabled [SampleMethod::LCM_SAMPLE_METHOD]. cfg_scale 2. 8 steps.
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
        .sampling_method(SampleMethod::LCM_SAMPLE_METHOD);
    Ok(builder)
}

/// Apply <https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/t5xxl_fp8_e4m3fn.safetensors> fp8_e4m3fn t5xxl text encoder to reduce memory usage
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

/// Apply <https://huggingface.co/kylielee505/mylcmlorassd> to reduce inference steps for SD v1 between 2-8 (default 8)
/// cfg_scale 1. 8 steps.
pub fn lcm_lora_ssd_1b(mut builder: ConfigsBuilder) -> Result<ConfigsBuilder, ApiError> {
    let lora_path = download_file_hf_hub(
        "kylielee505/mylcmlorassd",
        "pytorch_lora_weights.safetensors",
    )?;
    builder.1.lora_model(&lora_path);
    builder.0.cfg_scale(1.).steps(8);
    Ok(builder)
}

/// Enable vae tiling
pub fn vae_tiling(mut builder: ConfigsBuilder) -> Result<ConfigsBuilder, ApiError> {
    builder.1.vae_tiling(true);
    Ok(builder)
}

/// Enable preview with [crate::api::PreviewType::PREVIEW_PROJ]
pub fn preview_proj(mut builder: ConfigsBuilder) -> Result<ConfigsBuilder, ApiError> {
    builder.0.preview_mode(PreviewType::PREVIEW_PROJ);
    Ok(builder)
}

/// Enable preview with [crate::api::PreviewType::PREVIEW_TAE]
pub fn preview_tae(mut builder: ConfigsBuilder) -> Result<ConfigsBuilder, ApiError> {
    builder.0.preview_mode(PreviewType::PREVIEW_TAE);
    Ok(builder)
}

/// Enable preview with [crate::api::PreviewType::PREVIEW_VAE]
pub fn preview_vae(mut builder: ConfigsBuilder) -> Result<ConfigsBuilder, ApiError> {
    builder.0.preview_mode(PreviewType::PREVIEW_VAE);
    Ok(builder)
}

/// Enable easycache support with default values
pub fn enable_easycache(mut builder: ConfigsBuilder) -> Result<ConfigsBuilder, ApiError> {
    builder.1.easy_cache(true);
    Ok(builder)
}

/// Enable flash attention
pub fn enable_flash_attention(mut builder: ConfigsBuilder) -> Result<ConfigsBuilder, ApiError> {
    builder.1.flash_attention(true);
    Ok(builder)
}

#[cfg(test)]
mod tests {
    use crate::{
        api::gen_img,
        modifier::{
            enable_easycache, enable_flash_attention, lcm_lora_ssd_1b, offload_params_to_cpu,
            preview_proj, preview_tae, preview_vae, vae_tiling,
        },
        preset::{Flux1Weight, Modifier, Preset, PresetBuilder},
        util::set_hf_token,
    };

    use super::{
        hybrid_taesd, hybrid_taesd_xl, lcm_lora_sd_1_5, lcm_lora_sdxl_base_1_0, taesd, taesd_xl,
    };

    static PROMPT: &str = "a lovely dynosaur made by crochet";

    fn run(preset: Preset, m: Modifier) {
        let (config, model_config) = PresetBuilder::default()
            .preset(preset)
            .prompt(PROMPT)
            .with_modifier(m)
            .build()
            .unwrap();
        gen_img(config, model_config).unwrap();
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
            Preset::Flux1Schnell(Flux1Weight::Q2_K),
            offload_params_to_cpu,
        );
    }

    #[ignore]
    #[test]
    fn test_lcm_lora_ssd_1b() {
        run(
            Preset::SSD1B(crate::preset::SSD1BWeight::F8_E4M3),
            lcm_lora_ssd_1b,
        );
    }

    #[ignore]
    #[test]
    fn test_vae_tiling() {
        run(
            Preset::SSD1B(crate::preset::SSD1BWeight::F8_E4M3),
            vae_tiling,
        );
    }

    #[ignore]
    #[test]
    fn test_preview_proj() {
        run(Preset::SDXLTurbo1_0Fp16, preview_proj);
    }

    #[ignore]
    #[test]
    fn test_preview_tae() {
        run(Preset::SDXLTurbo1_0Fp16, preview_tae);
    }

    #[ignore]
    #[test]
    fn test_preview_vae() {
        run(Preset::SDXLTurbo1_0Fp16, preview_vae);
    }

    #[ignore]
    #[test]
    fn test_easy_cache() {
        set_hf_token(include_str!("../token.txt"));
        run(
            Preset::Flux1Mini(crate::preset::Flux1MiniWeight::Q2_K),
            enable_easycache,
        );
    }

    #[ignore]
    #[test]
    fn test_flash_attention() {
        set_hf_token(include_str!("../token.txt"));
        run(
            Preset::Flux1Mini(crate::preset::Flux1MiniWeight::Q2_K),
            enable_flash_attention,
        );
    }
}
