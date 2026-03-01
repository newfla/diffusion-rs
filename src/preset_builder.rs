use std::path::PathBuf;

use crate::{
    api::{ModelConfigBuilder, SampleMethod, Scheduler},
    modifier::{
        sdxl_vae_fp16_fix, t5xxl_fp16_flux_1, t5xxl_q2_k_flux_1, t5xxl_q3_k_flux_1,
        t5xxl_q4_k_flux_1, t5xxl_q8_0_flux_1,
    },
    preset::{
        AnimaWeight, ChromaRadianceWeight, ChromaWeight, ConfigsBuilder, DiffInstructStarWeight,
        Flux1MiniWeight, Flux1Weight, Flux2Klein4BWeight, Flux2Klein9BWeight,
        Flux2KleinBase4BWeight, Flux2KleinBase9BWeight, Flux2Weight, NitroSDRealismWeight,
        NitroSDVibrantWeight, OvisImageWeight, QwenImageWeight, SSD1BWeight,
        TwinFlowZImageTurboExpWeight, ZImageTurboWeight,
    },
};
use diffusion_rs_sys::scheduler_t;
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

pub fn stable_diffusion_3_medium() -> Result<ConfigsBuilder, ApiError> {
    let model_path = download_file_hf_hub(
        "stabilityai/stable-diffusion-3-medium",
        "sd3_medium_incl_clips_t5xxlfp16.safetensors",
    )?;

    let mut config = ConfigBuilder::default();

    config.cfg_scale(4.5).steps(30).height(1024).width(1024);

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

pub fn flux_1_dev(sd_type: Flux1Weight) -> Result<ConfigsBuilder, ApiError> {
    let model_path = flux_1_model_weight("dev", sd_type)?;
    let mut builder = flux_1_dev_schnell("dev", 28)?;

    builder.1.diffusion_model(model_path);
    match sd_type {
        Flux1Weight::Q4_0 => t5xxl_q4_k_flux_1(builder),
        Flux1Weight::Q8_0 => t5xxl_q8_0_flux_1(builder),
        Flux1Weight::Q2_K => t5xxl_q2_k_flux_1(builder),
        Flux1Weight::Q3_K => t5xxl_q3_k_flux_1(builder),
        Flux1Weight::Q4_K => t5xxl_q4_k_flux_1(builder),
    }
}

pub fn flux_1_schnell(sd_type: Flux1Weight) -> Result<ConfigsBuilder, ApiError> {
    let model_path = flux_1_model_weight("schnell", sd_type)?;
    let mut builder = flux_1_dev_schnell("schnell", 4)?;

    builder.1.diffusion_model(model_path);
    match sd_type {
        Flux1Weight::Q4_0 => t5xxl_q4_k_flux_1(builder),
        Flux1Weight::Q8_0 => t5xxl_q8_0_flux_1(builder),
        Flux1Weight::Q2_K => t5xxl_q2_k_flux_1(builder),
        Flux1Weight::Q3_K => t5xxl_q3_k_flux_1(builder),
        Flux1Weight::Q4_K => t5xxl_q4_k_flux_1(builder),
    }
}

fn flux_1_model_weight(model: &str, sd_type: Flux1Weight) -> Result<PathBuf, ApiError> {
    let weight_type = match sd_type {
        Flux1Weight::Q3_K => "q3_k",
        Flux1Weight::Q2_K => "q2_k",
        Flux1Weight::Q4_0 => "q4_0",
        Flux1Weight::Q4_K => "q4_k",
        Flux1Weight::Q8_0 => "q8_0",
    };
    download_file_hf_hub(
        format!("leejet/FLUX.1-{model}-gguf").as_str(),
        format!("flux1-{model}-{weight_type}.gguf").as_str(),
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
    config.cfg_scale(1.).steps(steps).height(1024).width(1024);

    Ok((config, model_config))
}

pub fn sd_turbo() -> Result<ConfigsBuilder, ApiError> {
    let model_path = download_file_hf_hub("stabilityai/sd-turbo", "sd_turbo.safetensors")?;

    let mut config = ConfigsBuilder::default();

    config.1.model(model_path);
    config.0.guidance(0.).cfg_scale(1.).steps(4);

    Ok(config)
}

pub fn sdxl_turbo_1_0() -> Result<ConfigsBuilder, ApiError> {
    let model_path =
        download_file_hf_hub("stabilityai/sdxl-turbo", "sd_xl_turbo_1.0_fp16.safetensors")?;

    let mut config = ConfigsBuilder::default();

    config.1.model(model_path);
    config.0.guidance(0.).cfg_scale(1.).steps(4);
    sdxl_vae_fp16_fix(config)
}

pub fn stable_diffusion_3_5_large() -> Result<ConfigsBuilder, ApiError> {
    stable_diffusion_3_5("large", "large", 28, 4.5)
}

pub fn stable_diffusion_3_5_large_turbo() -> Result<ConfigsBuilder, ApiError> {
    stable_diffusion_3_5("large-turbo", "large_turbo", 4, 0.)
}

pub fn stable_diffusion_3_5_medium() -> Result<ConfigsBuilder, ApiError> {
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
        .sampling_method(SampleMethod::EULER_SAMPLE_METHOD)
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
        .sampling_method(SampleMethod::DPM2_SAMPLE_METHOD)
        .steps(20)
        .guidance(6.)
        .height(1024)
        .width(1024);

    Ok(config)
}

pub fn flux_1_mini(sd_type: Flux1MiniWeight) -> Result<ConfigsBuilder, ApiError> {
    let model_path = flux_1_mini_model_weight(sd_type)?;
    let vae_path = download_file_hf_hub("Green-Sky/flux.1-schnell-GGUF", "ae-f16.gguf")?;
    let clip_l_path = download_file_hf_hub("Green-Sky/flux.1-schnell-GGUF", "clip_l-q8_0.gguf")?;
    let mut builder = flux_1_clip_vae(vae_path, clip_l_path, 20)?;
    builder.1.diffusion_model(model_path);
    builder.0.cfg_scale(1.);
    match sd_type {
        Flux1MiniWeight::F32 => t5xxl_fp16_flux_1(builder),
        Flux1MiniWeight::Q8_0 => t5xxl_q8_0_flux_1(builder),
        Flux1MiniWeight::Q2_K => t5xxl_q2_k_flux_1(builder),
        Flux1MiniWeight::Q3_K => t5xxl_q3_k_flux_1(builder),
        Flux1MiniWeight::Q5_K => t5xxl_q4_k_flux_1(builder),
        Flux1MiniWeight::Q6_K => t5xxl_q4_k_flux_1(builder),
        Flux1MiniWeight::BF16 => t5xxl_fp16_flux_1(builder),
    }
}

fn flux_1_mini_model_weight(sd_type: Flux1MiniWeight) -> Result<PathBuf, ApiError> {
    let (repo, file) = match sd_type {
        Flux1MiniWeight::F32 => ("TencentARC/flux-mini", "flux-mini.safetensors"),
        Flux1MiniWeight::BF16 => ("HyperX-Sentience/Flux-Mini-GGUF", "flux-mini-BF16.gguf"),
        Flux1MiniWeight::Q2_K => ("HyperX-Sentience/Flux-Mini-GGUF", "flux-mini-Q2_K.gguf"),
        Flux1MiniWeight::Q3_K => ("HyperX-Sentience/Flux-Mini-GGUF", "flux-mini-Q3_K.gguf"),
        Flux1MiniWeight::Q5_K => ("HyperX-Sentience/Flux-Mini-GGUF", "flux-mini-Q5_K.gguf"),
        Flux1MiniWeight::Q6_K => ("HyperX-Sentience/Flux-Mini-GGUF", "flux-mini-Q6_K.gguf"),
        Flux1MiniWeight::Q8_0 => ("HyperX-Sentience/Flux-Mini-GGUF", "flux-mini-Q8_0.gguf"),
    };
    download_file_hf_hub(repo, file)
}

pub fn chroma(sd_type: ChromaWeight) -> Result<ConfigsBuilder, ApiError> {
    let model_path = chroma_model_weight(sd_type)?;
    let vae_path = download_file_hf_hub("black-forest-labs/FLUX.1-dev", "ae.safetensors")?;
    let mut config = ConfigBuilder::default();
    let mut model_config = ModelConfigBuilder::default();

    model_config
        .diffusion_model(model_path)
        .vae(vae_path)
        .vae_tiling(true);
    config
        .cfg_scale(4.)
        .sampling_method(SampleMethod::EULER_SAMPLE_METHOD)
        .steps(20)
        .height(512)
        .width(512);

    let builder = (config, model_config);
    match sd_type {
        ChromaWeight::BF16 => t5xxl_fp16_flux_1(builder),
        ChromaWeight::Q4_0 => t5xxl_q4_k_flux_1(builder),
        _ => t5xxl_q8_0_flux_1(builder),
    }
}

fn chroma_model_weight(sd_type: ChromaWeight) -> Result<PathBuf, ApiError> {
    let (repo, file) = match sd_type {
        ChromaWeight::BF16 => (
            "silveroxides/Chroma-GGUF",
            "chroma-unlocked-v41/chroma-unlocked-v41-BF16.gguf",
        ),
        ChromaWeight::Q4_0 => (
            "silveroxides/Chroma-GGUF",
            "chroma-unlocked-v41/chroma-unlocked-v41-Q4_0.gguf",
        ),
        ChromaWeight::Q8_0 => (
            "silveroxides/Chroma-GGUF",
            "chroma-unlocked-v41/chroma-unlocked-v41-Q8_0.gguf",
        ),
    };
    download_file_hf_hub(repo, file)
}

pub fn nitro_sd_realism(sd_type: NitroSDRealismWeight) -> Result<ConfigsBuilder, ApiError> {
    let model_path = nitro_sd_realism_weight(sd_type)?;
    let mut config = ConfigBuilder::default();
    let mut model_config = ModelConfigBuilder::default();

    model_config
        .model(model_path)
        .timestep_shift(250)
        .scheduler(scheduler_t::SGM_UNIFORM_SCHEDULER);
    config.cfg_scale(1.).steps(1).height(1024).width(1024);
    Ok((config, model_config))
}

fn nitro_sd_realism_weight(sd_type: NitroSDRealismWeight) -> Result<PathBuf, ApiError> {
    let (repo, file) = match sd_type {
        NitroSDRealismWeight::F16 => ("mrfatso/NitroFusion-GGUF", "nitrosd-realism_f16.gguf"),
        NitroSDRealismWeight::Q2_K => ("mrfatso/NitroFusion-GGUF", "nitrosd-realism_q2_K.gguf"),
        NitroSDRealismWeight::Q3_K => ("mrfatso/NitroFusion-GGUF", "nitrosd-realism_q3_K.gguf"),
        NitroSDRealismWeight::Q4_0 => ("mrfatso/NitroFusion-GGUF", "nitrosd-realism_q4_0.gguf"),
        NitroSDRealismWeight::Q5_0 => ("mrfatso/NitroFusion-GGUF", "nitrosd-realism_q5_0.gguf"),
        NitroSDRealismWeight::Q6_K => ("mrfatso/NitroFusion-GGUF", "nitrosd-realism_q6_K.gguf"),
        NitroSDRealismWeight::Q8_0 => ("mrfatso/NitroFusion-GGUF", "nitrosd-realism_q8_0.gguf"),
    };
    download_file_hf_hub(repo, file)
}

pub fn nitro_sd_vibrant(sd_type: NitroSDVibrantWeight) -> Result<ConfigsBuilder, ApiError> {
    let model_path = nitro_sd_vibrant_weight(sd_type)?;
    let mut config = ConfigBuilder::default();
    let mut model_config = ModelConfigBuilder::default();

    model_config
        .model(model_path)
        .timestep_shift(500)
        .scheduler(scheduler_t::SGM_UNIFORM_SCHEDULER);
    config.cfg_scale(1.).steps(1).height(1024).width(1024);
    Ok((config, model_config))
}

fn nitro_sd_vibrant_weight(sd_type: NitroSDVibrantWeight) -> Result<PathBuf, ApiError> {
    let (repo, file) = match sd_type {
        NitroSDVibrantWeight::F16 => ("mrfatso/NitroFusion-GGUF", "nitrosd-vibrant_f16.gguf"),
        NitroSDVibrantWeight::Q2_K => ("mrfatso/NitroFusion-GGUF", "nitrosd-vibrant_q2_K.gguf"),
        NitroSDVibrantWeight::Q3_K => ("mrfatso/NitroFusion-GGUF", "nitrosd-vibrant_q3_K.gguf"),
        NitroSDVibrantWeight::Q4_0 => ("mrfatso/NitroFusion-GGUF", "nitrosd-vibrant_q4_0.gguf"),
        NitroSDVibrantWeight::Q5_0 => ("mrfatso/NitroFusion-GGUF", "nitrosd-vibrant_q5_0.gguf"),
        NitroSDVibrantWeight::Q6_K => ("mrfatso/NitroFusion-GGUF", "nitrosd-vibrant_q6_K.gguf"),
        NitroSDVibrantWeight::Q8_0 => ("mrfatso/NitroFusion-GGUF", "nitrosd-vibrant_q8_0.gguf"),
    };
    download_file_hf_hub(repo, file)
}

pub fn diff_instruct_star(sd_type: DiffInstructStarWeight) -> Result<ConfigsBuilder, ApiError> {
    let model_path = diff_instruct_star_weight(sd_type)?;
    let mut config = ConfigBuilder::default();
    let mut model_config = ModelConfigBuilder::default();

    model_config
        .model(model_path)
        .timestep_shift(400)
        .scheduler(scheduler_t::SGM_UNIFORM_SCHEDULER);
    config.cfg_scale(1.).steps(1).height(1024).width(1024);
    Ok((config, model_config))
}

fn diff_instruct_star_weight(sd_type: DiffInstructStarWeight) -> Result<PathBuf, ApiError> {
    let (repo, file) = match sd_type {
        DiffInstructStarWeight::F16 => (
            "mrfatso/Diff-InstructStar-GGUF",
            "Diff-InstructStar_f16.gguf",
        ),
        DiffInstructStarWeight::Q2_K => (
            "mrfatso/Diff-InstructStar-GGUF",
            "Diff-InstructStar_q2_K.gguf",
        ),
        DiffInstructStarWeight::Q3_K => (
            "mrfatso/Diff-InstructStar-GGUF",
            "Diff-InstructStar_q3_K.gguf",
        ),
        DiffInstructStarWeight::Q4_0 => (
            "mrfatso/Diff-InstructStar-GGUF",
            "Diff-InstructStar_q4_0.gguf",
        ),
        DiffInstructStarWeight::Q5_0 => (
            "mrfatso/Diff-InstructStar-GGUF",
            "Diff-InstructStar_q5_0.gguf",
        ),
        DiffInstructStarWeight::Q6_K => (
            "mrfatso/Diff-InstructStar-GGUF",
            "Diff-InstructStar_q6_K.gguf",
        ),
        DiffInstructStarWeight::Q8_0 => (
            "mrfatso/Diff-InstructStar-GGUF",
            "Diff-InstructStar_q8_0.gguf",
        ),
    };
    download_file_hf_hub(repo, file)
}

pub fn chroma_radiance(sd_type: ChromaRadianceWeight) -> Result<ConfigsBuilder, ApiError> {
    let model_path = chroma_radiance_weight(sd_type)?;
    let mut config = ConfigBuilder::default();
    let mut model_config = ModelConfigBuilder::default();

    model_config.model(model_path);
    config
        .cfg_scale(4.)
        .sampling_method(SampleMethod::EULER_SAMPLE_METHOD);
    t5xxl_fp16_flux_1((config, model_config))
}

fn chroma_radiance_weight(sd_type: ChromaRadianceWeight) -> Result<PathBuf, ApiError> {
    let (repo, file) = match sd_type {
        ChromaRadianceWeight::BF16 => (
            "silveroxides/Chroma1-Radiance-GGUF",
            "Chroma1-Radiance-v0.4/Chroma1-Radiance-v0.4-BF16.gguf",
        ),
        ChromaRadianceWeight::Q8_0 => (
            "silveroxides/Chroma1-Radiance-GGUF",
            "Chroma1-Radiance-v0.4/Chroma1-Radiance-v0.4-Q8_0.gguf",
        ),
    };
    download_file_hf_hub(repo, file)
}

pub fn ssd_1b(sd_type: SSD1BWeight) -> Result<ConfigsBuilder, ApiError> {
    let model = ssd_1b_weight(sd_type)?;
    let mut config = ConfigBuilder::default();
    let mut model_config = ModelConfigBuilder::default();

    model_config.model(model);
    config.cfg_scale(9.).height(1024).width(1024);
    Ok((config, model_config))
}

fn ssd_1b_weight(sd_type: SSD1BWeight) -> Result<PathBuf, ApiError> {
    let (repo, file) = match sd_type {
        SSD1BWeight::F16 => ("segmind/SSD-1B", "SSD-1B-A1111.safetensors"),
        SSD1BWeight::F8_E4M3 => (
            "hassenhamdi/SSD-1B-fp8_e4m3fn",
            "SSD-1B_fp8_e4m3fn.safetensors",
        ),
    };
    download_file_hf_hub(repo, file)
}

pub fn flux_2_dev(sd_type: Flux2Weight) -> Result<ConfigsBuilder, ApiError> {
    let (model, llm) = flux_2_dev_weight(sd_type)?;
    let vae = download_file_hf_hub(
        "black-forest-labs/FLUX.2-dev",
        "vae/diffusion_pytorch_model.safetensors",
    )?;
    let mut config = ConfigBuilder::default();
    let mut model_config = ModelConfigBuilder::default();

    model_config
        .diffusion_model(model)
        .llm(llm)
        .offload_params_to_cpu(true)
        .flash_attention(true)
        .vae(vae)
        .vae_tiling(true);
    config
        .cfg_scale(1.)
        .sampling_method(SampleMethod::EULER_SAMPLE_METHOD);

    Ok((config, model_config))
}

fn flux_2_dev_weight(sd_type: Flux2Weight) -> Result<(PathBuf, PathBuf), ApiError> {
    let (model, llm) = match sd_type {
        Flux2Weight::Q4_0 => (
            ("city96/FLUX.2-dev-gguf", "flux2-dev-Q4_0.gguf"),
            (
                "unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF",
                "Mistral-Small-3.2-24B-Instruct-2506-Q4_0.gguf",
            ),
        ),
        Flux2Weight::Q4_1 => (
            ("city96/FLUX.2-dev-gguf", "flux2-dev-Q4_1.gguf"),
            (
                "unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF",
                "Mistral-Small-3.2-24B-Instruct-2506-Q4_1.gguf",
            ),
        ),
        Flux2Weight::Q5_0 => (
            ("city96/FLUX.2-dev-gguf", "flux2-dev-Q5_0.gguf"),
            (
                "unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF",
                "Mistral-Small-3.2-24B-Instruct-2506-Q5_0.gguf",
            ),
        ),
        Flux2Weight::Q5_1 => (
            ("city96/FLUX.2-dev-gguf", "flux2-dev-Q5_1.gguf"),
            (
                "unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF",
                "Mistral-Small-3.2-24B-Instruct-2506-Q4_0.gguf",
            ),
        ),
        Flux2Weight::Q8_0 => (
            ("city96/FLUX.2-dev-gguf", "flux2-dev-Q8_0.gguf"),
            (
                "unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF",
                "Mistral-Small-3.2-24B-Instruct-2506-Q8_0.gguf",
            ),
        ),
        Flux2Weight::Q2_K => (
            ("city96/FLUX.2-dev-gguf", "flux2-dev-Q2_K.gguf"),
            (
                "unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF",
                "Mistral-Small-3.2-24B-Instruct-2506-Q2_K.gguf",
            ),
        ),
        Flux2Weight::Q3_K => (
            ("city96/FLUX.2-dev-gguf", "flux2-dev-Q3_K_M.gguf"),
            (
                "unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF",
                "Mistral-Small-3.2-24B-Instruct-2506-Q3_K_M.gguf",
            ),
        ),
        Flux2Weight::Q4_K => (
            ("city96/FLUX.2-dev-gguf", "flux2-dev-Q4_K_M.gguf"),
            (
                "unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF",
                "Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf",
            ),
        ),
        Flux2Weight::Q5_K => (
            ("city96/FLUX.2-dev-gguf", "flux2-dev-Q5_K_M.gguf"),
            (
                "unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF",
                "Mistral-Small-3.2-24B-Instruct-2506-Q5_K_M.gguf",
            ),
        ),
        Flux2Weight::Q6_K => (
            ("city96/FLUX.2-dev-gguf", "flux2-dev-Q6_K.gguf"),
            (
                "unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF",
                "Mistral-Small-3.2-24B-Instruct-2506-Q6_K.gguf",
            ),
        ),
        Flux2Weight::BF16 => (
            ("city96/FLUX.2-dev-gguf", "flux2-dev-BF16.gguf"),
            (
                "unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF",
                "Mistral-Small-3.2-24B-Instruct-2506-BF16.gguf",
            ),
        ),
    };
    let model_path = download_file_hf_hub(model.0, model.1)?;
    let llm_path = download_file_hf_hub(llm.0, llm.1)?;
    Ok((model_path, llm_path))
}

pub fn z_image_turbo(sd_type: ZImageTurboWeight) -> Result<ConfigsBuilder, ApiError> {
    let (model, llm) = z_image_turbo_weight(sd_type)?;
    let vae = download_file_hf_hub(
        "black-forest-labs/FLUX.1-schnell",
        "vae/diffusion_pytorch_model.safetensors",
    )?;
    let mut config = ConfigBuilder::default();
    let mut model_config = ModelConfigBuilder::default();

    model_config
        .diffusion_model(model)
        .llm(llm)
        .flash_attention(true)
        .vae(vae)
        .vae_tiling(true);
    config.steps(9).cfg_scale(1.).height(1024).width(512);

    Ok((config, model_config))
}

fn z_image_turbo_weight(sd_type: ZImageTurboWeight) -> Result<(PathBuf, PathBuf), ApiError> {
    let (model, llm) = match sd_type {
        ZImageTurboWeight::Q4_0 => (
            ("leejet/Z-Image-Turbo-GGUF", "z_image_turbo-Q4_0.gguf"),
            (
                "unsloth/Qwen3-4B-Instruct-2507-GGUF",
                "Qwen3-4B-Instruct-2507-Q4_0.gguf",
            ),
        ),
        ZImageTurboWeight::Q5_0 => (
            ("leejet/Z-Image-Turbo-GGUF", "z_image_turbo-Q5_0.gguf"),
            (
                "unsloth/Qwen3-4B-Instruct-2507-GGUF",
                "Qwen3-4B-Instruct-2507-Q4_0.gguf",
            ),
        ),
        ZImageTurboWeight::Q8_0 => (
            ("leejet/Z-Image-Turbo-GGUF", "z_image_turbo-Q8_0.gguf"),
            (
                "unsloth/Qwen3-4B-Instruct-2507-GGUF",
                "Qwen3-4B-Instruct-2507-Q8_0.gguf",
            ),
        ),
        ZImageTurboWeight::Q2_K => (
            ("leejet/Z-Image-Turbo-GGUF", "z_image_turbo-Q2_K.gguf"),
            (
                "unsloth/Qwen3-4B-Instruct-2507-GGUF",
                "Qwen3-4B-Instruct-2507-Q2_K.gguf",
            ),
        ),
        ZImageTurboWeight::Q3_K => (
            ("leejet/Z-Image-Turbo-GGUF", "z_image_turbo-Q3_K.gguf"),
            (
                "unsloth/Qwen3-4B-Instruct-2507-GGUF",
                "Qwen3-4B-Instruct-2507-Q3_K_M.gguf",
            ),
        ),
        ZImageTurboWeight::Q4_K => (
            ("leejet/Z-Image-Turbo-GGUF", "z_image_turbo-Q4_K.gguf"),
            (
                "unsloth/Qwen3-4B-Instruct-2507-GGUF",
                "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
            ),
        ),
        ZImageTurboWeight::Q6_K => (
            ("leejet/Z-Image-Turbo-GGUF", "z_image_turbo-Q6_0.gguf"),
            (
                "unsloth/Qwen3-4B-Instruct-2507-GGUF",
                "Qwen3-4B-Instruct-2507-Q6_K.gguf",
            ),
        ),
        ZImageTurboWeight::BF16 => (
            (
                "Comfy-Org/z_image_turbo",
                "split_files/diffusion_models/z_image_turbo_bf16.safetensors",
            ),
            (
                "unsloth/Qwen3-4B-Instruct-2507-GGUF",
                "Qwen3-4B-Instruct-2507-F16.gguf",
            ),
        ),
    };
    let model_path = download_file_hf_hub(model.0, model.1)?;
    let llm_path = download_file_hf_hub(llm.0, llm.1)?;
    Ok((model_path, llm_path))
}

pub fn qwen_image(sd_type: QwenImageWeight) -> Result<ConfigsBuilder, ApiError> {
    let (model, llm) = qwen_image_weight(sd_type)?;
    let vae = download_file_hf_hub(
        "Comfy-Org/Qwen-Image_ComfyUI",
        "split_files/vae/qwen_image_vae.safetensors",
    )?;
    let mut config = ConfigBuilder::default();
    let mut model_config = ModelConfigBuilder::default();

    model_config
        .diffusion_model(model)
        .llm(llm)
        .vae(vae)
        .offload_params_to_cpu(true)
        .flash_attention(true)
        .vae_tiling(true)
        .flow_shift(3.0);
    config
        .sampling_method(SampleMethod::EULER_SAMPLE_METHOD)
        .cfg_scale(2.5)
        .height(1024)
        .width(1024);

    Ok((config, model_config))
}

fn qwen_image_weight(sd_type: QwenImageWeight) -> Result<(PathBuf, PathBuf), ApiError> {
    let (model, llm) = match sd_type {
        QwenImageWeight::Q4_0 => (
            ("QuantStack/Qwen-Image-GGUF", "Qwen_Image-Q4_0.gguf"),
            (
                "mradermacher/Qwen2.5-VL-7B-Instruct-GGUF",
                "Qwen2.5-VL-7B-Instruct.Q4_K_M.gguf",
            ),
        ),
        QwenImageWeight::Q4_1 => (
            ("QuantStack/Qwen-Image-GGUF", "Qwen_Image-Q4_1.gguf"),
            (
                "mradermacher/Qwen2.5-VL-7B-Instruct-GGUF",
                "Qwen2.5-VL-7B-Instruct.Q4_K_M.gguf",
            ),
        ),
        QwenImageWeight::Q5_0 => (
            ("QuantStack/Qwen-Image-GGUF", "Qwen_Image-Q5_0.gguf"),
            (
                "mradermacher/Qwen2.5-VL-7B-Instruct-GGUF",
                "Qwen2.5-VL-7B-Instruct.Q5_K_M.gguf",
            ),
        ),
        QwenImageWeight::Q5_1 => (
            ("QuantStack/Qwen-Image-GGUF", "Qwen_Image-Q5_1.gguf"),
            (
                "mradermacher/Qwen2.5-VL-7B-Instruct-GGUF",
                "Qwen2.5-VL-7B-Instruct.Q5_K_M.gguf",
            ),
        ),
        QwenImageWeight::Q8_0 => (
            ("QuantStack/Qwen-Image-GGUF", "Qwen_Image-Q8_0.gguf"),
            (
                "mradermacher/Qwen2.5-VL-7B-Instruct-GGUF",
                "Qwen2.5-VL-7B-Instruct.Q8_0.gguf",
            ),
        ),
        QwenImageWeight::Q2_K => (
            ("QuantStack/Qwen-Image-GGUF", "Qwen_Image-Q2_K.gguf"),
            (
                "mradermacher/Qwen2.5-VL-7B-Instruct-GGUF",
                "Qwen2.5-VL-7B-Instruct.Q2_K.gguf",
            ),
        ),
        QwenImageWeight::Q3_K => (
            ("QuantStack/Qwen-Image-GGUF", "Qwen_Image-Q3_K_M.gguf"),
            (
                "mradermacher/Qwen2.5-VL-7B-Instruct-GGUF",
                "Qwen2.5-VL-7B-Instruct.Q3_K_M.gguf",
            ),
        ),
        QwenImageWeight::Q4_K => (
            ("QuantStack/Qwen-Image-GGUF", "Qwen_Image-Q4_K_M.gguf"),
            (
                "mradermacher/Qwen2.5-VL-7B-Instruct-GGUF",
                "Qwen2.5-VL-7B-Instruct.Q4_K_M.gguf",
            ),
        ),
        QwenImageWeight::Q5_K => (
            ("QuantStack/Qwen-Image-GGUF", "Qwen_Image-Q5_K_M.gguf"),
            (
                "mradermacher/Qwen2.5-VL-7B-Instruct-GGUF",
                "Qwen2.5-VL-7B-Instruct.Q5_K_M.gguf",
            ),
        ),
        QwenImageWeight::Q6_K => (
            ("QuantStack/Qwen-Image-GGUF", "Qwen_Image-Q6_K.gguf"),
            (
                "mradermacher/Qwen2.5-VL-7B-Instruct-GGUF",
                "Qwen2.5-VL-7B-Instruct.Q6_K.gguf",
            ),
        ),
        QwenImageWeight::BF16 => (
            (
                "Comfy-Org/Qwen-Image_ComfyUI",
                "split_files/diffusion_models/qwen_image_bf16.safetensors",
            ),
            (
                "Comfy-Org/Qwen-Image_ComfyUI",
                "split_files/text_encoders/qwen_2.5_vl_7b.safetensors",
            ),
        ),
        QwenImageWeight::F8_E4M3 => (
            (
                "Comfy-Org/Qwen-Image_ComfyUI",
                "split_files/diffusion_models/qwen_image_fp8_e4m3fn.safetensors",
            ),
            (
                "Comfy-Org/Qwen-Image_ComfyUI",
                "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
            ),
        ),
    };
    let model_path = download_file_hf_hub(model.0, model.1)?;
    let llm_path = download_file_hf_hub(llm.0, llm.1)?;
    Ok((model_path, llm_path))
}

pub fn ovis_image(sd_type: OvisImageWeight) -> Result<ConfigsBuilder, ApiError> {
    let (model, llm) = ovis_image_weight(sd_type)?;
    let vae = download_file_hf_hub(
        "black-forest-labs/FLUX.1-schnell",
        "vae/diffusion_pytorch_model.safetensors",
    )?;
    let mut config = ConfigBuilder::default();
    let mut model_config = ModelConfigBuilder::default();

    model_config
        .diffusion_model(model)
        .llm(llm)
        .vae(vae)
        .offload_params_to_cpu(true)
        .flash_attention(true)
        .vae_tiling(true);
    config.steps(20).cfg_scale(5.).height(512).width(512);

    Ok((config, model_config))
}

fn ovis_image_weight(sd_type: OvisImageWeight) -> Result<(PathBuf, PathBuf), ApiError> {
    let model = match sd_type {
        OvisImageWeight::Q4_0 => ("leejet/Ovis-Image-7B-GGUF", "ovis_image-Q4_0.gguf"),
        OvisImageWeight::Q8_0 => ("leejet/Ovis-Image-7B-GGUF", "ovis_image-Q8_0.gguf"),
        OvisImageWeight::BF16 => (
            "Comfy-Org/Ovis-Image",
            "split_files/diffusion_models/ovis_image_bf16.safetensors",
        ),
    };
    let model_path = download_file_hf_hub(model.0, model.1)?;
    let llm_path = download_file_hf_hub(
        "Comfy-Org/Ovis-Image",
        "split_files/text_encoders/ovis_2.5.safetensors",
    )?;
    Ok((model_path, llm_path))
}

pub fn dream_shaper_xl_2_1_turbo() -> Result<ConfigsBuilder, ApiError> {
    let model_path = download_file_hf_hub(
        "Lykon/dreamshaper-xl-v2-turbo",
        "DreamShaperXL_Turbo_v2_1.safetensors",
    )?;

    let mut config = ConfigsBuilder::default();

    config.1.model(model_path).vae_tiling(true);
    config
        .0
        .sampling_method(SampleMethod::DPM2_SAMPLE_METHOD)
        .steps(6)
        .guidance(2.)
        .height(1024)
        .width(1024);

    Ok(config)
}

pub fn twinflow_z_image_turbo(
    sd_type: TwinFlowZImageTurboExpWeight,
) -> Result<ConfigsBuilder, ApiError> {
    let (model, llm) = twinflow_z_image_turbo_weight(sd_type)?;
    let vae = download_file_hf_hub(
        "black-forest-labs/FLUX.1-schnell",
        "vae/diffusion_pytorch_model.safetensors",
    )?;
    let mut config = ConfigBuilder::default();
    let mut model_config = ModelConfigBuilder::default();

    model_config
        .diffusion_model(model)
        .llm(llm)
        .flash_attention(true)
        .vae(vae)
        .vae_tiling(true)
        .scheduler(Scheduler::SMOOTHSTEP_SCHEDULER);
    config
        .steps(3)
        .cfg_scale(1.)
        .height(1024)
        .width(512)
        .sampling_method(SampleMethod::DPM2_SAMPLE_METHOD);

    Ok((config, model_config))
}

fn twinflow_z_image_turbo_weight(
    sd_type: TwinFlowZImageTurboExpWeight,
) -> Result<(PathBuf, PathBuf), ApiError> {
    let (model, llm) = match sd_type {
        TwinFlowZImageTurboExpWeight::Q4_0 => (
            (
                "wbruna/TwinFlow-Z-Image-Turbo-sdcpp-GGUF",
                "TwinFlow_Z_Image_Turbo_exp-Q4_0.gguf",
            ),
            (
                "unsloth/Qwen3-4B-Instruct-2507-GGUF",
                "Qwen3-4B-Instruct-2507-Q4_0.gguf",
            ),
        ),
        TwinFlowZImageTurboExpWeight::Q5_0 => (
            (
                "wbruna/TwinFlow-Z-Image-Turbo-sdcpp-GGUF",
                "TwinFlow_Z_Image_Turbo_exp-Q5_0.gguf",
            ),
            (
                "unsloth/Qwen3-4B-Instruct-2507-GGUF",
                "Qwen3-4B-Instruct-2507-Q4_0.gguf",
            ),
        ),
        TwinFlowZImageTurboExpWeight::Q8_0 => (
            (
                "wbruna/TwinFlow-Z-Image-Turbo-sdcpp-GGUF",
                "TwinFlow_Z_Image_Turbo_exp-Q8_0.gguf",
            ),
            (
                "unsloth/Qwen3-4B-Instruct-2507-GGUF",
                "Qwen3-4B-Instruct-2507-Q8_0.gguf",
            ),
        ),
        TwinFlowZImageTurboExpWeight::Q3_K => (
            (
                "wbruna/TwinFlow-Z-Image-Turbo-sdcpp-GGUF",
                "TwinFlow_Z_Image_Turbo_exp-Q3_K.gguf",
            ),
            (
                "unsloth/Qwen3-4B-Instruct-2507-GGUF",
                "Qwen3-4B-Instruct-2507-Q3_K_M.gguf",
            ),
        ),
        TwinFlowZImageTurboExpWeight::Q6_K => (
            (
                "wbruna/TwinFlow-Z-Image-Turbo-sdcpp-GGUF",
                "TwinFlow_Z_Image_Turbo_exp-Q3_K.gguf",
            ),
            (
                "unsloth/Qwen3-4B-Instruct-2507-GGUF",
                "Qwen3-4B-Instruct-2507-Q6_K.gguf",
            ),
        ),
        TwinFlowZImageTurboExpWeight::BF16 => (
            (
                "azazeal2/TwinFlow-Z-Image-Turbo-repacked",
                "TwinFlow_Z_Image_Turbo_exp_bf16.safetensors",
            ),
            (
                "unsloth/Qwen3-4B-Instruct-2507-GGUF",
                "Qwen3-4B-Instruct-2507-F16.gguf",
            ),
        ),
    };
    let model_path = download_file_hf_hub(model.0, model.1)?;
    let llm_path = download_file_hf_hub(llm.0, llm.1)?;
    Ok((model_path, llm_path))
}

pub fn sdxs512_dream_shaper() -> Result<ConfigsBuilder, ApiError> {
    let model = download_file_hf_hub("akleine/sdxs-512", "sdxs.safetensors")?;
    let mut config = ConfigBuilder::default();
    let mut model_config = ModelConfigBuilder::default();

    model_config.model(model);
    config.steps(1).cfg_scale(1.).height(512).width(512);

    Ok((config, model_config))
}

pub fn flux_2_klein_4b(sd_type: Flux2Klein4BWeight) -> Result<ConfigsBuilder, ApiError> {
    let (model, llm) = flux_2_klein_4b_weight(sd_type)?;
    let vae = download_file_hf_hub(
        "black-forest-labs/FLUX.2-dev",
        "vae/diffusion_pytorch_model.safetensors",
    )?;
    let mut config = ConfigBuilder::default();
    let mut model_config = ModelConfigBuilder::default();

    model_config
        .diffusion_model(model)
        .llm(llm)
        .vae(vae)
        .offload_params_to_cpu(true)
        .flash_attention(true)
        .vae_tiling(true);
    config.cfg_scale(1.).steps(4).height(1024).width(1024);

    Ok((config, model_config))
}

fn flux_2_klein_4b_weight(sd_type: Flux2Klein4BWeight) -> Result<(PathBuf, PathBuf), ApiError> {
    let (model, llm) = match sd_type {
        Flux2Klein4BWeight::Q4_0 => (
            ("leejet/FLUX.2-klein-4B-GGUF", "flux-2-klein-4b-Q4_0.gguf"),
            ("unsloth/Qwen3-4B-GGUF", "Qwen3-4B-Q4_0.gguf"),
        ),
        Flux2Klein4BWeight::Q8_0 => (
            ("leejet/FLUX.2-klein-4B-GGUF", "flux-2-klein-4b-Q8_0.gguf"),
            ("unsloth/Qwen3-4B-GGUF", "Qwen3-4B-Q8_0.gguf"),
        ),
        Flux2Klein4BWeight::BF16 => (
            (
                "black-forest-labs/FLUX.2-klein-4B",
                "flux-2-klein-4b.safetensors",
            ),
            ("unsloth/Qwen3-4B-GGUF", "Qwen3-4B-BF16.gguf"),
        ),
    };
    let model_path = download_file_hf_hub(model.0, model.1)?;
    let llm_path = download_file_hf_hub(llm.0, llm.1)?;
    Ok((model_path, llm_path))
}

pub fn flux_2_klein_base_4b(sd_type: Flux2KleinBase4BWeight) -> Result<ConfigsBuilder, ApiError> {
    let (model, llm) = flux_2_klein_base_4b_weight(sd_type)?;
    let vae = download_file_hf_hub(
        "black-forest-labs/FLUX.2-dev",
        "vae/diffusion_pytorch_model.safetensors",
    )?;
    let mut config = ConfigBuilder::default();
    let mut model_config = ModelConfigBuilder::default();

    model_config
        .diffusion_model(model)
        .llm(llm)
        .vae(vae)
        .offload_params_to_cpu(true)
        .flash_attention(true)
        .vae_tiling(true);
    config.cfg_scale(4.).steps(20).height(1024).width(1024);

    Ok((config, model_config))
}

fn flux_2_klein_base_4b_weight(
    sd_type: Flux2KleinBase4BWeight,
) -> Result<(PathBuf, PathBuf), ApiError> {
    let (model, llm) = match sd_type {
        Flux2KleinBase4BWeight::Q4_0 => (
            (
                "leejet/FLUX.2-klein-base-4B-GGUF",
                "flux-2-klein-base-4b-Q4_0.gguf",
            ),
            ("unsloth/Qwen3-4B-GGUF", "Qwen3-4B-Q4_0.gguf"),
        ),
        Flux2KleinBase4BWeight::Q8_0 => (
            (
                "leejet/FLUX.2-klein-base-4B-GGUF",
                "flux-2-klein-base-4b-Q8_0.gguf",
            ),
            ("unsloth/Qwen3-4B-GGUF", "Qwen3-4B-Q8_0.gguf"),
        ),
        Flux2KleinBase4BWeight::BF16 => (
            (
                "black-forest-labs/FLUX.2-klein-base-4B",
                "flux-2-klein-base-4b.safetensors",
            ),
            ("unsloth/Qwen3-4B-GGUF", "Qwen3-4B-BF16.gguf"),
        ),
    };
    let model_path = download_file_hf_hub(model.0, model.1)?;
    let llm_path = download_file_hf_hub(llm.0, llm.1)?;
    Ok((model_path, llm_path))
}

pub fn flux_2_klein_9b(sd_type: Flux2Klein9BWeight) -> Result<ConfigsBuilder, ApiError> {
    let (model, llm) = flux_2_klein_9b_weight(sd_type)?;
    let vae = download_file_hf_hub(
        "black-forest-labs/FLUX.2-dev",
        "vae/diffusion_pytorch_model.safetensors",
    )?;
    let mut config = ConfigBuilder::default();
    let mut model_config = ModelConfigBuilder::default();

    model_config
        .diffusion_model(model)
        .llm(llm)
        .vae(vae)
        .offload_params_to_cpu(true)
        .flash_attention(true)
        .vae_tiling(true);
    config.cfg_scale(1.).steps(4).height(1024).width(1024);

    Ok((config, model_config))
}

fn flux_2_klein_9b_weight(sd_type: Flux2Klein9BWeight) -> Result<(PathBuf, PathBuf), ApiError> {
    let (model, llm) = match sd_type {
        Flux2Klein9BWeight::Q4_0 => (
            ("leejet/FLUX.2-klein-9B-GGUF", "flux-2-klein-9b-Q4_0.gguf"),
            ("unsloth/Qwen3-8B-GGUF", "Qwen3-8B-Q4_K_M.gguf"),
        ),
        Flux2Klein9BWeight::Q8_0 => (
            ("leejet/FLUX.2-klein-9B-GGUF", "flux-2-klein-9b-Q8_0.gguf"),
            ("unsloth/Qwen3-8B-GGUF", "Qwen3-8B-Q8_0.gguf"),
        ),
        Flux2Klein9BWeight::BF16 => (
            (
                "black-forest-labs/FLUX.2-klein-9B",
                "flux-2-klein-9b.safetensors",
            ),
            ("unsloth/Qwen3-8B-GGUF", "Qwen3-8B-BF16.gguf"),
        ),
    };
    let model_path = download_file_hf_hub(model.0, model.1)?;
    let llm_path = download_file_hf_hub(llm.0, llm.1)?;
    Ok((model_path, llm_path))
}

pub fn flux_2_klein_base_9b(sd_type: Flux2KleinBase9BWeight) -> Result<ConfigsBuilder, ApiError> {
    let (model, llm) = flux_2_klein_base_9b_weight(sd_type)?;
    let vae = download_file_hf_hub(
        "black-forest-labs/FLUX.2-dev",
        "vae/diffusion_pytorch_model.safetensors",
    )?;
    let mut config = ConfigBuilder::default();
    let mut model_config = ModelConfigBuilder::default();

    model_config
        .diffusion_model(model)
        .llm(llm)
        .vae(vae)
        .offload_params_to_cpu(true)
        .flash_attention(true)
        .vae_tiling(true);
    config.cfg_scale(4.).steps(20).height(1024).width(1024);

    Ok((config, model_config))
}

fn flux_2_klein_base_9b_weight(
    sd_type: Flux2KleinBase9BWeight,
) -> Result<(PathBuf, PathBuf), ApiError> {
    let (model, llm) = match sd_type {
        Flux2KleinBase9BWeight::Q4_0 => (
            (
                "leejet/FLUX.2-klein-base-9B-GGUF",
                "flux-2-klein-base-9b-Q4_0.gguf",
            ),
            ("unsloth/Qwen3-8B-GGUF", "Qwen3-8B-Q4_K_M.gguf"),
        ),
        Flux2KleinBase9BWeight::BF16 => (
            (
                "black-forest-labs/FLUX.2-klein-base-9B",
                "flux-2-klein-base-9b.safetensors",
            ),
            ("unsloth/Qwen3-8B-GGUF", "Qwen3-8B-BF16.gguf"),
        ),
    };
    let model_path = download_file_hf_hub(model.0, model.1)?;
    let llm_path = download_file_hf_hub(llm.0, llm.1)?;
    Ok((model_path, llm_path))
}

pub fn segmind_vega() -> Result<ConfigsBuilder, ApiError> {
    let model = download_file_hf_hub("segmind/Segmind-Vega", "segmind-vega.safetensors")?;
    let mut config = ConfigBuilder::default();
    let mut model_config = ModelConfigBuilder::default();

    model_config.model(model).vae_tiling(true);
    config.guidance(9.).steps(25).height(1024).width(1024);

    Ok((config, model_config))
}

pub fn anima(sd_type: AnimaWeight) -> Result<ConfigsBuilder, ApiError> {
    let (model, llm) = anima_weight(sd_type)?;
    let vae = download_file_hf_hub(
        "circlestone-labs/Anima",
        "split_files/vae/qwen_image_vae.safetensors",
    )?;
    let mut config = ConfigBuilder::default();
    let mut model_config = ModelConfigBuilder::default();

    model_config
        .diffusion_model(model)
        .llm(llm)
        .vae(vae)
        .vae_tiling(true);
    config.cfg_scale(4.).steps(30).height(1024).width(1024);

    Ok((config, model_config))
}

fn anima_weight(sd_type: AnimaWeight) -> Result<(PathBuf, PathBuf), ApiError> {
    let (model, llm) = match sd_type {
        AnimaWeight::Q4_K => (
            ("Bedovyy/Anima-GGUF", "anima-preview-Q4_K_M.gguf"),
            (
                "mradermacher/Qwen3-0.6B-Base-GGUF",
                "Qwen3-0.6B-Base.Q4_K_M.gguf",
            ),
        ),
        AnimaWeight::Q5_K => (
            ("Bedovyy/Anima-GGUF", "anima-preview-Q5_K_M.gguf"),
            (
                "mradermacher/Qwen3-0.6B-Base-GGUF",
                "Qwen3-0.6B-Base.Q5_K_M.gguf",
            ),
        ),
        AnimaWeight::Q6_K => (
            ("Bedovyy/Anima-GGUF", "anima-preview-Q6_K.gguf"),
            (
                "mradermacher/Qwen3-0.6B-Base-GGUF",
                "Qwen3-0.6B-Base.Q6_K.gguf",
            ),
        ),
        AnimaWeight::BF16 => (
            (
                "circlestone-labs/Anima",
                "split_files/diffusion_models/anima-preview.safetensors",
            ),
            (
                "circlestone-labs/Anima",
                "split_files/text_encoders/qwen_3_06b_base.safetensors",
            ),
        ),
        AnimaWeight::Q4_0 => (
            ("Bedovyy/Anima-GGUF", "anima-preview-Q4_0.gguf"),
            (
                "mradermacher/Qwen3-0.6B-Base-GGUF",
                "Qwen3-0.6B-Base.Q4_K_M.gguf",
            ),
        ),
        AnimaWeight::Q4_1 => (
            ("Bedovyy/Anima-GGUF", "anima-preview-Q4_1.gguf"),
            (
                "mradermacher/Qwen3-0.6B-Base-GGUF",
                "Qwen3-0.6B-Base.Q4_K_M.gguf",
            ),
        ),
        AnimaWeight::Q5_0 => (
            ("Bedovyy/Anima-GGUF", "anima-preview-Q5_0.gguf"),
            (
                "mradermacher/Qwen3-0.6B-Base-GGUF",
                "Qwen3-0.6B-Base.Q5_K_M.gguf",
            ),
        ),
        AnimaWeight::Q5_1 => (
            ("Bedovyy/Anima-GGUF", "anima-preview-Q5_1.gguf"),
            (
                "mradermacher/Qwen3-0.6B-Base-GGUF",
                "Qwen3-0.6B-Base.Q5_K_M.gguf",
            ),
        ),
        AnimaWeight::Q8_0 => (
            ("Bedovyy/Anima-GGUF", "anima-preview-Q8_0.gguf"),
            (
                "mradermacher/Qwen3-0.6B-Base-GGUF",
                "Qwen3-0.6B-Base.Q8_0.gguf",
            ),
        ),
        AnimaWeight::Q3_K => (
            ("Bedovyy/Anima-GGUF", "anima-preview-Q3_K_L.gguf"),
            (
                "mradermacher/Qwen3-0.6B-Base-GGUF",
                "Qwen3-0.6B-Base.Q3_K_L.gguf",
            ),
        ),
    };
    let model_path = download_file_hf_hub(model.0, model.1)?;
    let llm_path = download_file_hf_hub(llm.0, llm.1)?;
    Ok((model_path, llm_path))
}
