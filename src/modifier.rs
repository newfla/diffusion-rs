use hf_hub::api::sync::ApiError;

use crate::{api::ConfigBuilder, util::download_file_hf_hub};

/// Add the <https://huggingface.co/ximso/RealESRGAN_x4plus_anime_6B> upscaler
pub fn real_esrgan_x4plus_anime_6_b(mut builder: ConfigBuilder) -> Result<ConfigBuilder, ApiError> {
    let upscaler_path = download_file_hf_hub(
        "ximso/RealESRGAN_x4plus_anime_6B",
        "RealESRGAN_x4plus_anime_6B.pth",
    )?;
    builder.upscale_model(upscaler_path);
    Ok(builder)
}

/// Apply <https://huggingface.co/madebyollin/sdxl-vae-fp16-fix> to avoid black images with xl models
pub fn sdxl_vae_fp16_fix(mut builder: ConfigBuilder) -> Result<ConfigBuilder, ApiError> {
    let vae_path = download_file_hf_hub("madebyollin/sdxl-vae-fp16-fix", "sdxl.vae.safetensors")?;
    builder.vae(vae_path);
    Ok(builder)
}

/// Apply <https://huggingface.co/madebyollin/taesd> taesd autoencoder for faster decoding (SD v1/v2)
pub fn taesd(mut builder: ConfigBuilder) -> Result<ConfigBuilder, ApiError> {
    let taesd_path =
        download_file_hf_hub("madebyollin/taesd", "diffusion_pytorch_model.safetensors")?;
    builder.taesd(taesd_path);
    Ok(builder)
}

/// Apply <https://huggingface.co/madebyollin/taesdxl> taesd autoencoder for faster decoding (SDXL)
pub fn taesd_xl(mut builder: ConfigBuilder) -> Result<ConfigBuilder, ApiError> {
    let taesd_path =
        download_file_hf_hub("madebyollin/taesdxl", "diffusion_pytorch_model.safetensors")?;
    builder.taesd(taesd_path);
    Ok(builder)
}

#[cfg(test)]
mod tests {
    use crate::{
        api::txt2img,
        preset::{Modifier, Preset, PresetBuilder},
    };

    use super::{taesd, taesd_xl};

    static PROMPT: &str = "a lovely duck drinking water from a bottle";

    fn run(preset: Preset, m: Modifier) {
        let config = PresetBuilder::default()
            .preset(preset)
            .prompt(PROMPT)
            .with_modifier(m)
            .build()
            .unwrap();
        txt2img(config).unwrap();
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
}
