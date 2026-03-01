use hf_hub::api::sync::ApiError;

use crate::{
    api::{LoraSpec, PreviewType, SampleMethod},
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
    builder.1.lora_models(
        lora_path.parent().unwrap(),
        vec![LoraSpec {
            file_name: "pytorch_lora_weights".to_string(),
            is_high_noise: false,
            multiplier: 1.0,
        }],
    );
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

    builder.1.lora_models(
        lora_path.parent().unwrap(),
        vec![LoraSpec {
            file_name: "pytorch_lora_weights".to_string(),
            is_high_noise: false,
            multiplier: 1.0,
        }],
    );
    builder
        .0
        .cfg_scale(2.)
        .steps(8)
        .sampling_method(SampleMethod::LCM_SAMPLE_METHOD);
    Ok(builder)
}

/// Apply <https://huggingface.co/nerijs/pixel-art-xl>
pub fn lora_pixel_art_sdxl_base_1_0(
    mut builder: ConfigsBuilder,
) -> Result<ConfigsBuilder, ApiError> {
    let lora_path = download_file_hf_hub("nerijs/pixel-art-xl", "pixel-art-xl.safetensors")?;

    builder.1.lora_models(
        lora_path.parent().unwrap(),
        vec![LoraSpec {
            file_name: "pixel-art-xl".to_string(),
            is_high_noise: false,
            multiplier: 1.2,
        }],
    );
    Ok(builder)
}

/// Apply <https://huggingface.co/nerijs/pastelcomic-flux>
pub fn lora_pastelcomic_2_flux(mut builder: ConfigsBuilder) -> Result<ConfigsBuilder, ApiError> {
    let lora_path = download_file_hf_hub("nerijs/pastelcomic-flux", "pastelcomic_v2.safetensors")?;

    builder.1.lora_models(
        lora_path.parent().unwrap(),
        vec![LoraSpec {
            file_name: "pastelcomic_v2".to_string(),
            is_high_noise: false,
            multiplier: 1.2,
        }],
    );
    Ok(builder)
}

/// Apply <https://huggingface.co/strangerzonehf/Ghibli-Flux-Cartoon-LoRA>
pub fn lora_ghibli_flux(mut builder: ConfigsBuilder) -> Result<ConfigsBuilder, ApiError> {
    let lora_path = download_file_hf_hub(
        "strangerzonehf/Ghibli-Flux-Cartoon-LoRA",
        "Ghibili-Cartoon-Art.safetensors",
    )?;

    builder.1.lora_models(
        lora_path.parent().unwrap(),
        vec![LoraSpec {
            file_name: "Ghibili-Cartoon-Art".to_string(),
            is_high_noise: false,
            multiplier: 1.0,
        }],
    );
    Ok(builder)
}

/// Apply <https://huggingface.co/strangerzonehf/Flux-Midjourney-Mix2-LoRA>
pub fn lora_midjourney_mix_2_flux(mut builder: ConfigsBuilder) -> Result<ConfigsBuilder, ApiError> {
    let lora_path = download_file_hf_hub(
        "strangerzonehf/Flux-Midjourney-Mix2-LoRA",
        "mjV6.safetensors",
    )?;

    builder.1.lora_models(
        lora_path.parent().unwrap(),
        vec![LoraSpec {
            file_name: "mjV6".to_string(),
            is_high_noise: false,
            multiplier: 1.0,
        }],
    );
    Ok(builder)
}

/// Apply <https://huggingface.co/prithivMLmods/Retro-Pixel-Flux-LoRA>
pub fn lora_retro_pixel_flux(mut builder: ConfigsBuilder) -> Result<ConfigsBuilder, ApiError> {
    let lora_path = download_file_hf_hub(
        "prithivMLmods/Retro-Pixel-Flux-LoRA",
        "Retro-Pixel.safetensors",
    )?;

    builder.1.lora_models(
        lora_path.parent().unwrap(),
        vec![LoraSpec {
            file_name: "Retro-Pixel".to_string(),
            is_high_noise: false,
            multiplier: 1.0,
        }],
    );
    Ok(builder)
}

/// Apply <https://huggingface.co/prithivMLmods/Canopus-Pixar-3D-Flux-LoRA>
pub fn lora_canopus_pixar_3d_flux(mut builder: ConfigsBuilder) -> Result<ConfigsBuilder, ApiError> {
    let lora_path = download_file_hf_hub(
        "prithivMLmods/Canopus-Pixar-3D-Flux-LoRA",
        "Canopus-Pixar-3D-FluxDev-LoRA.safetensors",
    )?;

    builder.1.lora_models(
        lora_path.parent().unwrap(),
        vec![LoraSpec {
            file_name: "Canopus-Pixar-3D-FluxDev-LoRA".to_string(),
            is_high_noise: false,
            multiplier: 1.0,
        }],
    );
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

/// Offload model parameters to CPU (for low VRAM GPUs)
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
    builder.1.lora_models(
        lora_path.parent().unwrap(),
        vec![LoraSpec {
            file_name: "pytorch_lora_weights".to_string(),
            is_high_noise: false,
            multiplier: 1.0,
        }],
    );
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

/// Enable flash attention
pub fn enable_flash_attention(mut builder: ConfigsBuilder) -> Result<ConfigsBuilder, ApiError> {
    builder.1.flash_attention(true);
    Ok(builder)
}

/// Apply <https://huggingface.co/segmind/Segmind-VegaRT> to [crate::preset::Preset::SegmindVega]
pub fn lcm_lora_segmind_vega_rt(mut builder: ConfigsBuilder) -> Result<ConfigsBuilder, ApiError> {
    let lora_path =
        download_file_hf_hub("segmind/Segmind-VegaRT", "pytorch_lora_weights.safetensors")?;
    builder.1.lora_models(
        lora_path.parent().unwrap(),
        vec![LoraSpec {
            file_name: "pytorch_lora_weights".to_string(),
            is_high_noise: false,
            multiplier: 1.0,
        }],
    );
    builder.0.guidance(0.).steps(4);
    Ok(builder)
}

/// Apply <https://huggingface.co/Einhorn/Anima-Preview_8_Step_Turbo_Lora>
pub fn lora_anima_8_steps_turbo(mut builder: ConfigsBuilder) -> Result<ConfigsBuilder, ApiError> {
    let lora_path = download_file_hf_hub(
        "Einhorn/Anima-Preview_8_Step_Turbo_Lora",
        "Anima-Preview_Turbo_8step.safetensors",
    )?;

    builder.1.lora_models(
        lora_path.parent().unwrap(),
        vec![LoraSpec {
            file_name: "Anima-Preview_Turbo_8step".to_string(),
            is_high_noise: false,
            multiplier: 1.0,
        }],
    );
    builder.0.cfg_scale(1.).steps(8);
    Ok(builder)
}

#[cfg(test)]
mod tests {
    use hf_hub::api::sync::ApiError;

    use crate::{
        api::gen_img,
        modifier::{
            enable_flash_attention, lcm_lora_segmind_vega_rt, lcm_lora_ssd_1b,
            lora_anima_8_steps_turbo, lora_canopus_pixar_3d_flux, lora_ghibli_flux,
            lora_midjourney_mix_2_flux, lora_pastelcomic_2_flux, lora_pixel_art_sdxl_base_1_0,
            lora_retro_pixel_flux, offload_params_to_cpu, preview_proj, preview_tae, preview_vae,
            vae_tiling,
        },
        preset::{AnimaWeight, ConfigsBuilder, Flux1Weight, Preset, PresetBuilder},
        util::set_hf_token,
    };

    use super::{
        hybrid_taesd, hybrid_taesd_xl, lcm_lora_sd_1_5, lcm_lora_sdxl_base_1_0, taesd, taesd_xl,
    };

    static PROMPT: &str = "a lovely dinosaur made by crochet";

    fn run<F>(preset: Preset, prompt: &str, m: F)
    where
        F: FnOnce(ConfigsBuilder) -> Result<ConfigsBuilder, ApiError> + 'static,
    {
        let (mut config, mut model_config) = PresetBuilder::default()
            .preset(preset)
            .prompt(prompt)
            .with_modifier(m)
            .build()
            .unwrap();
        gen_img(&mut config, &mut model_config).unwrap();
    }

    #[ignore]
    #[test]
    fn test_taesd() {
        run(Preset::StableDiffusion1_5, PROMPT, taesd);
    }

    #[ignore]
    #[test]
    fn test_taesd_xl() {
        run(Preset::SDXLTurbo1_0, PROMPT, taesd_xl);
    }

    #[ignore]
    #[test]
    fn test_hybrid_taesd() {
        run(Preset::StableDiffusion1_5, PROMPT, hybrid_taesd);
    }

    #[ignore]
    #[test]
    fn test_hybrid_taesd_xl() {
        run(Preset::SDXLTurbo1_0, PROMPT, hybrid_taesd_xl);
    }

    #[ignore]
    #[test]
    fn test_lcm_lora_sd_1_5() {
        run(Preset::StableDiffusion1_5, PROMPT, lcm_lora_sd_1_5);
    }

    #[ignore]
    #[test]
    fn test_lcm_lora_sdxl_base_1_0() {
        run(Preset::SDXLBase1_0, PROMPT, lcm_lora_sdxl_base_1_0);
    }

    #[ignore]
    #[test]
    fn test_offload_params_to_cpu() {
        set_hf_token(include_str!("../token.txt"));
        run(
            Preset::Flux1Schnell(Flux1Weight::Q2_K),
            PROMPT,
            offload_params_to_cpu,
        );
    }

    #[ignore]
    #[test]
    fn test_lcm_lora_ssd_1b() {
        run(
            Preset::SSD1B(crate::preset::SSD1BWeight::F8_E4M3),
            PROMPT,
            lcm_lora_ssd_1b,
        );
    }

    #[ignore]
    #[test]
    fn test_vae_tiling() {
        run(
            Preset::SSD1B(crate::preset::SSD1BWeight::F8_E4M3),
            PROMPT,
            vae_tiling,
        );
    }

    #[ignore]
    #[test]
    fn test_preview_proj() {
        run(Preset::SDXLTurbo1_0, PROMPT, preview_proj);
    }

    #[ignore]
    #[test]
    fn test_preview_tae() {
        run(Preset::SDXLTurbo1_0, PROMPT, preview_tae);
    }

    #[ignore]
    #[test]
    fn test_preview_vae() {
        run(Preset::SDXLTurbo1_0, PROMPT, preview_vae);
    }

    #[ignore]
    #[test]
    fn test_flash_attention() {
        set_hf_token(include_str!("../token.txt"));
        run(
            Preset::Flux1Mini(crate::preset::Flux1MiniWeight::Q2_K),
            PROMPT,
            enable_flash_attention,
        );
    }

    #[ignore]
    #[test]
    fn test_segmind_vega_rt_lcm_lora() {
        run(Preset::SegmindVega, PROMPT, lcm_lora_segmind_vega_rt);
    }

    #[ignore]
    #[test]
    fn test_lora_pixel_art_xl() {
        run(
            Preset::SDXLBase1_0,
            "pixel, a cute corgi",
            lora_pixel_art_sdxl_base_1_0,
        );
    }

    #[ignore]
    #[test]
    fn test_lora_pastelcomic_2_flux() {
        set_hf_token(include_str!("../token.txt"));
        run(
            Preset::Flux1Schnell(Flux1Weight::Q2_K),
            PROMPT,
            lora_pastelcomic_2_flux,
        );
    }

    #[ignore]
    #[test]
    fn test_lora_ghibli_flux() {
        set_hf_token(include_str!("../token.txt"));
        run(
            Preset::Flux1Schnell(Flux1Weight::Q2_K),
            "Ghibli Art – A wise old fisherman sits on a wooden dock, gazing out at the vast, blue ocean. He wears a worn-out straw hat and a navy-blue coat, and he holds a fishing rod in his hands. A black cat with bright green eyes sits beside him, watching the waves. In the distance, a lighthouse stands tall against the horizon, with seagulls soaring in the sky. The water glistens under the golden sunset.",
            lora_ghibli_flux,
        );
    }

    #[ignore]
    #[test]
    fn test_lora_midjourney_mix_2_flux() {
        set_hf_token(include_str!("../token.txt"));
        run(
            Preset::Flux1Schnell(Flux1Weight::Q2_K),
            "MJ v6, delicious dipped chocolate pastry japo gallery, white background, in the style of dark brown, close-up intensity, duckcore, rounded, high resolution --ar 2:3 --v 5",
            lora_midjourney_mix_2_flux,
        );
    }

    #[ignore]
    #[test]
    fn test_lora_retro_pixel_flux() {
        set_hf_token(include_str!("../token.txt"));
        run(
            Preset::Flux1Schnell(Flux1Weight::Q2_K),
            "Retro Pixel, pixel art of a Hamburger in the style of an old video game, hero, pixelated 8bit, final boss ",
            lora_retro_pixel_flux,
        );
    }

    #[ignore]
    #[test]
    fn test_lora_canopus_pixar_3d_flux() {
        set_hf_token(include_str!("../token.txt"));
        run(
            Preset::Flux1Schnell(Flux1Weight::Q2_K),
            "A young man with light brown wavy hair and light brown eyes sitting in an armchair and looking directly at the camera, pixar style, disney pixar, office background, ultra detailed, 1 man",
            lora_canopus_pixar_3d_flux,
        );
    }

    #[ignore]
    #[test]
    fn test_lora_anima_8_steps_turbo() {
        run(
            Preset::Anima(AnimaWeight::Q6_K),
            PROMPT,
            lora_anima_8_steps_turbo,
        );
    }
}
