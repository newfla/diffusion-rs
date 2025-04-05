use std::path::PathBuf;

use crate::{
    api::{self, ModelConfigBuilder, SampleMethod},
    modifier::{sdxl_vae_fp16_fix, t5xxl_fp16_flux_1, t5xxl_q2_k_flux_1},
    preset::ConfigsBuilder,
};
use hf_hub::api::sync::ApiError;

use crate::{api::ConfigBuilder, util::download_file_hf_hub};

pub fn stable_diffusion_1_4() -> Result<ConfigsBuilder, ApiError> {
    let model_path =
        download_file_hf_hub("CompVis/stable-diffusion-v-1-4-original", "sd-v1-4.ckpt")?;

    let mut model_config = ModelConfigBuilder::default();

    model_config.model(model_path);

    Ok((ConfigBuilder::default(), model_config))
}

pub fn stable_diffusion_1_5() -> Result<ConfigsBuilder, ApiError> {
    let model_path = download_file_hf_hub(
        "stablediffusiontutorials/stable-diffusion-v1.5",
        "v1-5-pruned-emaonly.safetensors",
    )?;

    let mut model_config = ModelConfigBuilder::default();

    model_config.model(model_path);

    Ok((ConfigBuilder::default(), model_config))
}

pub fn stable_diffusion_2_1() -> Result<ConfigsBuilder, ApiError> {
    let model_path = download_file_hf_hub(
        "stabilityai/stable-diffusion-2-1",
        "v2-1_768-nonema-pruned.safetensors",
    )?;

    let mut config = ConfigBuilder::default();

    config.steps(25).height(768).width(768);

    let mut model_config = ModelConfigBuilder::default();

    model_config.model(model_path).vae_tiling(true);

    Ok((config, model_config))
}

pub fn stable_diffusion_3_medium_fp16() -> Result<ConfigsBuilder, ApiError> {
    let model_path = download_file_hf_hub(
        "stabilityai/stable-diffusion-3-medium",
        "sd3_medium_incl_clips_t5xxlfp16.safetensors",
    )?;

    let mut config = ConfigBuilder::default();

    config
        .cfg_scale(4.5)
        .sampling_method(SampleMethod::EULER)
        .steps(30)
        .height(1024)
        .width(1024);

    let mut model_config = ModelConfigBuilder::default();

    model_config.model(model_path).vae_tiling(true);

    Ok((config, model_config))
}

pub fn sdxl_base_1_0() -> Result<ConfigsBuilder, ApiError> {
    let model_path = download_file_hf_hub(
        "stabilityai/stable-diffusion-xl-base-1.0",
        "sd_xl_base_1.0.safetensors",
    )?;

    let mut config = ConfigBuilder::default();

    config.height(1024).width(1024);

    let mut model_config = ModelConfigBuilder::default();

    model_config.model(model_path).vae_tiling(true);
    sdxl_vae_fp16_fix((config, model_config))
}

pub fn flux_1_dev(sd_type: api::WeightType) -> Result<ConfigsBuilder, ApiError> {
    let model_path = flux_1_model_weight("dev", sd_type)?;
    let mut builder = flux_1_dev_schnell("dev", 28)?;

    builder.1.diffusion_model(model_path);
    t5xxl_fp16_flux_1(builder)
}

pub fn flux_1_schnell(sd_type: api::WeightType) -> Result<ConfigsBuilder, ApiError> {
    let model_path = flux_1_model_weight("schnell", sd_type)?;
    let mut builder = flux_1_dev_schnell("schnell", 4)?;

    builder.1.diffusion_model(model_path);
    t5xxl_fp16_flux_1(builder)
}

fn flux_1_model_weight(model: &str, sd_type: api::WeightType) -> Result<PathBuf, ApiError> {
    check_flux_1_type(sd_type);
    let weight_type = flux_1_type_to_model(sd_type);
    download_file_hf_hub(
        format!("leejet/FLUX.1-{model}-gguf").as_str(),
        format!("flux1-{model}-{}.gguf", weight_type).as_str(),
    )
}

fn flux_1_dev_schnell(vae_model: &str, steps: i32) -> Result<ConfigsBuilder, ApiError> {
    let vae_path = download_file_hf_hub(
        format!("black-forest-labs/FLUX.1-{vae_model}").as_str(),
        "ae.safetensors",
    )?;
    let clip_l_path =
        download_file_hf_hub("comfyanonymous/flux_text_encoders", "clip_l.safetensors")?;

    flux_1_clip_vae(vae_path, clip_l_path, steps)
}

fn flux_1_clip_vae(
    vae_path: PathBuf,
    clip_l_path: PathBuf,
    steps: i32,
) -> Result<ConfigsBuilder, ApiError> {
    let mut config = ConfigBuilder::default();
    let mut model_config = ModelConfigBuilder::default();

    model_config
        .vae(vae_path)
        .clip_l(clip_l_path)
        .vae_tiling(true);
    config
        .cfg_scale(1.)
        .sampling_method(SampleMethod::EULER)
        .steps(steps)
        .height(1024)
        .width(1024);

    Ok((config, model_config))
}

fn check_flux_1_type(sd_type: api::WeightType) {
    assert!(
        sd_type == api::WeightType::SD_TYPE_Q2_K
            || sd_type == api::WeightType::SD_TYPE_Q3_K
            || sd_type == api::WeightType::SD_TYPE_Q4_0
            || sd_type == api::WeightType::SD_TYPE_Q4_K
            || sd_type == api::WeightType::SD_TYPE_Q8_0
    );
}

fn flux_1_type_to_model(sd_type: api::WeightType) -> &'static str {
    match sd_type {
        api::WeightType::SD_TYPE_Q3_K => "q3_k",
        api::WeightType::SD_TYPE_Q2_K => "q2_k",
        api::WeightType::SD_TYPE_Q4_0 => "q4_0",
        api::WeightType::SD_TYPE_Q4_K => "q4_k",
        api::WeightType::SD_TYPE_Q8_0 => "q8_0",
        _ => "not_supported",
    }
}

pub fn sd_turbo() -> Result<ConfigsBuilder, ApiError> {
    let model_path = download_file_hf_hub("stabilityai/sd-turbo", "sd_turbo.safetensors")?;

    let mut config = ConfigsBuilder::default();

    config.1.model(model_path);
    config.0.guidance(0.).cfg_scale(1.).steps(4);

    Ok(config)
}

pub fn sdxl_turbo_1_0_fp16() -> Result<ConfigsBuilder, ApiError> {
    let model_path =
        download_file_hf_hub("stabilityai/sdxl-turbo", "sd_xl_turbo_1.0_fp16.safetensors")?;

    let mut config = ConfigsBuilder::default();

    config.1.model(model_path);
    config.0.guidance(0.).cfg_scale(1.).steps(4);
    sdxl_vae_fp16_fix(config)
}

pub fn stable_diffusion_3_5_large_fp16() -> Result<ConfigsBuilder, ApiError> {
    stable_diffusion_3_5("large", "large", 28, 4.5)
}

pub fn stable_diffusion_3_5_large_turbo_fp16() -> Result<ConfigsBuilder, ApiError> {
    stable_diffusion_3_5("large-turbo", "large_turbo", 4, 0.)
}

pub fn stable_diffusion_3_5_medium_fp16() -> Result<ConfigsBuilder, ApiError> {
    stable_diffusion_3_5("medium", "medium", 40, 4.5)
}

pub fn stable_diffusion_3_5(
    model: &str,
    file_model: &str,
    steps: i32,
    cfg_scale: f32,
) -> Result<ConfigsBuilder, ApiError> {
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

    let mut config = ConfigsBuilder::default();

    config
        .1
        .diffusion_model(model_path)
        .clip_l(clip_l_path)
        .clip_g(clip_g_path)
        .t5xxl(t5xxl_path)
        .vae_tiling(true);
    config
        .0
        .cfg_scale(cfg_scale)
        .sampling_method(SampleMethod::EULER)
        .steps(steps)
        .height(1024)
        .width(1024);

    Ok(config)
}

pub fn juggernaut_xl_11() -> Result<ConfigsBuilder, ApiError> {
    let model_path = download_file_hf_hub(
        "RunDiffusion/Juggernaut-XI-v11",
        "Juggernaut-XI-byRunDiffusion.safetensors",
    )?;

    let mut config = ConfigsBuilder::default();

    config.1.model(model_path).vae_tiling(true);
    config
        .0
        .sampling_method(SampleMethod::DPM2)
        .steps(20)
        .guidance(6.)
        .height(1024)
        .width(1024);

    Ok(config)
}

pub fn flux_1_mini(sd_type: api::WeightType) -> Result<ConfigsBuilder, ApiError> {
    let model_path = flux_1_mini_model_weight(sd_type)?;
    let vae_path = download_file_hf_hub("Green-Sky/flux.1-schnell-GGUF", "ae-f16.gguf")?;
    let clip_l_path = download_file_hf_hub("Green-Sky/flux.1-schnell-GGUF", "clip_l-q8_0.gguf")?;
    let mut builder = flux_1_clip_vae(vae_path, clip_l_path, 20)?;
    builder.1.diffusion_model(model_path);
    builder.0.cfg_scale(1.);
    t5xxl_q2_k_flux_1(builder)
}

fn flux_1_mini_model_weight(sd_type: api::WeightType) -> Result<PathBuf, ApiError> {
    check_flux_1_mini_type(sd_type);
    flux_1_mini_type_to_model(sd_type)
}

fn check_flux_1_mini_type(sd_type: api::WeightType) {
    assert!(
        sd_type == api::WeightType::SD_TYPE_F32
            || sd_type == api::WeightType::SD_TYPE_BF16
            || sd_type == api::WeightType::SD_TYPE_Q2_K
            || sd_type == api::WeightType::SD_TYPE_Q3_K
            || sd_type == api::WeightType::SD_TYPE_Q5_K
            || sd_type == api::WeightType::SD_TYPE_Q6_K
            || sd_type == api::WeightType::SD_TYPE_Q8_0
    );
}

fn flux_1_mini_type_to_model(sd_type: api::WeightType) -> Result<PathBuf, ApiError> {
    let (repo, file) = match sd_type {
        api::WeightType::SD_TYPE_F32 => ("TencentARC/flux-mini", "flux-mini.safetensors"),
        api::WeightType::SD_TYPE_BF16 => ("HyperX-Sentience/Flux-Mini-GGUF", "flux-mini-BF16.gguf"),
        api::WeightType::SD_TYPE_Q2_K => ("HyperX-Sentience/Flux-Mini-GGUF", "flux-mini-Q2_K.gguf"),
        api::WeightType::SD_TYPE_Q3_K => ("HyperX-Sentience/Flux-Mini-GGUF", "flux-mini-Q3_K.gguf"),
        api::WeightType::SD_TYPE_Q5_K => ("HyperX-Sentience/Flux-Mini-GGUF", "flux-mini-Q5_K.gguf"),
        api::WeightType::SD_TYPE_Q6_K => ("HyperX-Sentience/Flux-Mini-GGUF", "flux-mini-Q6_K.gguf"),
        api::WeightType::SD_TYPE_Q8_0 => ("HyperX-Sentience/Flux-Mini-GGUF", "flux-mini-Q8_0.gguf"),
        _ => ("not_supported", "not_supported"),
    };
    download_file_hf_hub(repo, file)
}
