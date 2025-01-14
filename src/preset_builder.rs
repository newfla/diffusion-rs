use std::path::PathBuf;

use crate::{
    api::{self, SampleMethod},
    modifier::{sdxl_vae_fp16_fix, t5xxl_fp16_flux_1, t5xxl_fp8_flux_1},
};
use hf_hub::api::sync::ApiError;

use crate::{api::ConfigBuilder, util::download_file_hf_hub};

pub fn stable_diffusion_1_4() -> Result<ConfigBuilder, ApiError> {
    let model_path =
        download_file_hf_hub("CompVis/stable-diffusion-v-1-4-original", "sd-v1-4.ckpt")?;

    let mut config = ConfigBuilder::default();

    config.model(model_path);

    Ok(config)
}

pub fn stable_diffusion_1_5() -> Result<ConfigBuilder, ApiError> {
    let model_path = download_file_hf_hub(
        "stablediffusiontutorials/stable-diffusion-v1.5",
        "v1-5-pruned-emaonly.safetensors",
    )?;

    let mut config = ConfigBuilder::default();

    config.model(model_path);

    Ok(config)
}

pub fn stable_diffusion_2_1() -> Result<ConfigBuilder, ApiError> {
    let model_path = download_file_hf_hub(
        "stabilityai/stable-diffusion-2-1",
        "v2-1_768-nonema-pruned.safetensors",
    )?;

    let mut config = ConfigBuilder::default();

    config
        .model(model_path)
        .vae_tiling(true)
        .steps(25)
        .height(768)
        .width(768);

    Ok(config)
}

pub fn stable_diffusion_3_medium_fp16() -> Result<ConfigBuilder, ApiError> {
    let model_path = download_file_hf_hub(
        "stabilityai/stable-diffusion-3-medium",
        "sd3_medium_incl_clips_t5xxlfp16.safetensors",
    )?;

    let mut config = ConfigBuilder::default();

    config
        .model(model_path)
        .vae_tiling(true)
        .cfg_scale(4.5)
        .sampling_method(SampleMethod::EULER)
        .steps(30)
        .height(1024)
        .width(1024);

    Ok(config)
}

pub fn sdxl_base_1_0() -> Result<ConfigBuilder, ApiError> {
    let model_path = download_file_hf_hub(
        "stabilityai/stable-diffusion-xl-base-1.0",
        "sd_xl_base_1.0.safetensors",
    )?;

    let mut config = ConfigBuilder::default();

    config
        .model(model_path)
        .vae_tiling(true)
        .height(1024)
        .width(1024);
    sdxl_vae_fp16_fix(config)
}

pub fn flux_1_dev(sd_type: api::WeightType) -> Result<ConfigBuilder, ApiError> {
    let model_path = flux_1_model_weight("dev", sd_type)?;
    let mut builder = flux_1("dev", 28)?;

    builder.diffusion_model(model_path);
    t5xxl_fp16_flux_1(builder)
}

pub fn flux_1_schnell(sd_type: api::WeightType) -> Result<ConfigBuilder, ApiError> {
    let model_path = flux_1_model_weight("schnell", sd_type)?;
    let mut builder = flux_1("schnell", 4)?;

    builder.diffusion_model(model_path);
    t5xxl_fp16_flux_1(builder)
}

fn flux_1_model_weight(model: &str, sd_type: api::WeightType) -> Result<PathBuf, ApiError> {
    check_flux_type(sd_type);
    let weight_type = flux_type_to_model(sd_type);
    download_file_hf_hub(
        format!("leejet/FLUX.1-{model}-gguf").as_str(),
        format!("flux1-{model}-{}.gguf", weight_type).as_str(),
    )
}

fn flux_1(vae_model: &str, steps: i32) -> Result<ConfigBuilder, ApiError> {
    let mut config = ConfigBuilder::default();
    let vae_path = download_file_hf_hub(
        format!("black-forest-labs/FLUX.1-{vae_model}").as_str(),
        "ae.safetensors",
    )?;
    let clip_l_path =
        download_file_hf_hub("comfyanonymous/flux_text_encoders", "clip_l.safetensors")?;

    config
        .vae(vae_path)
        .clip_l(clip_l_path)
        .vae_tiling(true)
        .cfg_scale(1.)
        .sampling_method(SampleMethod::EULER)
        .steps(steps)
        .height(1024)
        .width(1024);

    Ok(config)
}

fn check_flux_type(sd_type: api::WeightType) {
    assert!(
        sd_type == api::WeightType::SD_TYPE_Q2_K
            || sd_type == api::WeightType::SD_TYPE_Q3_K
            || sd_type == api::WeightType::SD_TYPE_Q4_0
            || sd_type == api::WeightType::SD_TYPE_Q4_K
            || sd_type == api::WeightType::SD_TYPE_Q8_0
    );
}

fn flux_type_to_model(sd_type: api::WeightType) -> &'static str {
    match sd_type {
        api::WeightType::SD_TYPE_Q3_K => "q3_k",
        api::WeightType::SD_TYPE_Q2_K => "q2_k",
        api::WeightType::SD_TYPE_Q4_0 => "q4_0",
        api::WeightType::SD_TYPE_Q4_K => "q4_k",
        api::WeightType::SD_TYPE_Q8_0 => "q8_0",
        _ => "not_supported",
    }
}

pub fn sd_turbo() -> Result<ConfigBuilder, ApiError> {
    let model_path = download_file_hf_hub("stabilityai/sd-turbo", "sd_turbo.safetensors")?;

    let mut config = ConfigBuilder::default();

    config.model(model_path).guidance(0.).cfg_scale(1.).steps(4);

    Ok(config)
}

pub fn sdxl_turbo_1_0_fp16() -> Result<ConfigBuilder, ApiError> {
    let model_path =
        download_file_hf_hub("stabilityai/sdxl-turbo", "sd_xl_turbo_1.0_fp16.safetensors")?;

    let mut config = ConfigBuilder::default();

    config.model(model_path).guidance(0.).cfg_scale(1.).steps(4);
    sdxl_vae_fp16_fix(config)
}

pub fn stable_diffusion_3_5_large_fp16() -> Result<ConfigBuilder, ApiError> {
    stable_diffusion_3_5("large", "large", 28, 4.5)
}

pub fn stable_diffusion_3_5_large_turbo_fp16() -> Result<ConfigBuilder, ApiError> {
    stable_diffusion_3_5("large-turbo", "large_turbo", 4, 0.)
}

pub fn stable_diffusion_3_5_medium_fp16() -> Result<ConfigBuilder, ApiError> {
    stable_diffusion_3_5("medium", "medium", 40, 4.5)
}

pub fn stable_diffusion_3_5(
    model: &str,
    file_model: &str,
    steps: i32,
    cfg_scale: f32,
) -> Result<ConfigBuilder, ApiError> {
    let model_path = download_file_hf_hub(
        format!("stabilityai/stable-diffusion-3.5-{model}").as_str(),
        format!("sd3.5_{file_model}.safetensors").as_str(),
    )?;

    let clip_g_path = download_file_hf_hub(
        "Comfy-Org/stable-diffusion-3.5-fp8",
        "text_encoders/clip_g.safetensors",
    )?;
    let clip_l_path = download_file_hf_hub(
        "Comfy-Org/stable-diffusion-3.5-fp8",
        "text_encoders/clip_l.safetensors",
    )?;
    let t5xxl_path = download_file_hf_hub(
        "Comfy-Org/stable-diffusion-3.5-fp8",
        "text_encoders/t5xxl_fp16.safetensors",
    )?;

    let mut config = ConfigBuilder::default();

    config
        .diffusion_model(model_path)
        .clip_l(clip_l_path)
        .clip_g(clip_g_path)
        .t5xxl(t5xxl_path)
        .vae_tiling(true)
        .cfg_scale(cfg_scale)
        .sampling_method(SampleMethod::EULER)
        .steps(steps)
        .height(1024)
        .width(1024);

    Ok(config)
}

pub fn juggernaut_xl_11() -> Result<ConfigBuilder, ApiError> {
    let model_path = download_file_hf_hub(
        "RunDiffusion/Juggernaut-XI-v11",
        "Juggernaut-XI-byRunDiffusion.safetensors",
    )?;

    let mut config = ConfigBuilder::default();

    config
        .model(model_path)
        .vae_tiling(true)
        .sampling_method(SampleMethod::DPM2)
        .steps(20)
        .guidance(6.)
        .height(1024)
        .width(1024);

    Ok(config)
}

pub fn flux_1_mini() -> Result<ConfigBuilder, ApiError> {
    let model_path = download_file_hf_hub("TencentARC/flux-mini", "flux-mini.safetensors")?;
    let mut builder = flux_1("dev", 28)?;
    builder.diffusion_model(model_path).width(512).height(512);
    t5xxl_fp8_flux_1(builder)
}
