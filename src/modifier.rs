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
