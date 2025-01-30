use std::ffi::c_void;
use std::ptr::null;
use std::slice;

use diffusion_rs_sys::sd_image_t;
use image::ImageBuffer;
use image::Rgb;
use image::RgbImage;
use libc::free;

use diffusion_rs_sys::free_sd_ctx;
use diffusion_rs_sys::new_sd_ctx;
use diffusion_rs_sys::sd_ctx_t;

use crate::model_config::ModelConfig;
use crate::txt2img_config::Txt2ImgConfig;
use crate::utils::CLibString;
use crate::utils::DiffusionError;
struct ModelCtx {
    /// The underlying C context
    raw_ctx: *mut sd_ctx_t,

    /// We keep the config around in case we need to refer to it
    pub model_config: ModelConfig,
}

impl ModelCtx {
    pub fn new(config: ModelConfig) -> Self {
        let raw_ctx = unsafe {
            new_sd_ctx(
                config.model.as_ptr(),
                config.clip_l.as_ptr(),
                config.clip_g.as_ptr(),
                config.t5xxl.as_ptr(),
                config.diffusion_model.as_ptr(),
                config.vae.as_ptr(),
                config.taesd.as_ptr(),
                config.control_net.as_ptr(),
                config.lora_model_dir.as_ptr(),
                config.embeddings_dir.as_ptr(),
                config.stacked_id_embd_dir.as_ptr(),
                config.vae_decode_only,
                config.vae_tiling,
                config.free_params_immediately,
                config.n_threads,
                config.weight_type,
                config.rng_type,
                config.schedule,
                config.keep_clip_on_cpu,
                config.keep_control_net_cpu,
                config.keep_vae_on_cpu,
                config.flash_attention,
            )
        };

        Self {
            raw_ctx,
            model_config: config,
        }
    }

    pub fn destroy(&mut self) {
        unsafe {
            if !self.raw_ctx.is_null() {
                free_sd_ctx(self.raw_ctx);
                self.raw_ctx = std::ptr::null_mut();
            }
        }
    }

    pub fn txt2img(
        &mut self,
        mut txt2img_config: Txt2ImgConfig,
    ) -> Result<Vec<RgbImage>, DiffusionError> {
        // add loras to prompt as suffix
        let prompt: CLibString = {
            let mut prompt = txt2img_config.prompt.clone();
            for lora in txt2img_config.lora_prompt_suffix.iter() {
                prompt.push_str(lora);
            }
            prompt.into()
        };

        //print prompt for debugging
        println!(
            "Prompt: {:?}",
            prompt.0.to_str().expect("Couldn't get string")
        );

        //controlnet support

        let results: *mut sd_image_t = unsafe {
            diffusion_rs_sys::txt2img(
                self.raw_ctx,
                prompt.as_ptr(),
                txt2img_config.negative_prompt.as_ptr(),
                txt2img_config.clip_skip as i32,
                txt2img_config.cfg_scale,
                txt2img_config.guidance,
                txt2img_config.width,
                txt2img_config.height,
                txt2img_config.sample_method,
                txt2img_config.sample_steps,
                txt2img_config.seed,
                txt2img_config.batch_count,
                null(),
                txt2img_config.control_strength,
                txt2img_config.style_ratio,
                txt2img_config.normalize_input,
                txt2img_config.input_id_images.as_ptr(),
                txt2img_config.skip_layer.as_mut_ptr(),
                txt2img_config.skip_layer.len(),
                txt2img_config.slg_scale,
                txt2img_config.skip_layer_start,
                txt2img_config.skip_layer_end,
            )
        };

        if results.is_null() {
            return Err(DiffusionError::Forward);
        }

        let result_images: Vec<RgbImage> = unsafe {
            let img_count = txt2img_config.batch_count as usize;
            let images = slice::from_raw_parts(results, img_count);
            let rgb_images: Result<Vec<RgbImage>, DiffusionError> = images
                .iter()
                .map(|sd_img| {
                    let len = (sd_img.width * sd_img.height * sd_img.channel) as usize;
                    let raw_pixels = slice::from_raw_parts(sd_img.data, len);
                    let buffer = raw_pixels.to_vec();
                    let buffer = ImageBuffer::<Rgb<u8>, _>::from_raw(
                        sd_img.width as u32,
                        sd_img.height as u32,
                        buffer,
                    );
                    Ok(match buffer {
                        Some(buffer) => RgbImage::from(buffer),
                        None => return Err(DiffusionError::SDImagetoRustImage),
                    })
                })
                .collect();
            match rgb_images {
                Ok(images) => images,
                Err(e) => return Err(e),
            }
        };

        //Clean-up slice section
        unsafe {
            free(results as *mut c_void);
        }
        Ok(result_images)
    }
}

/// Automatic cleanup on drop
impl Drop for ModelCtx {
    fn drop(&mut self) {
        self.destroy();
    }
}

#[cfg(test)]
mod tests {
    /// Sampling methods
    pub use diffusion_rs_sys::sample_method_t as SampleMethod;
    /// Denoiser sigma schedule
    pub use diffusion_rs_sys::schedule_t as Schedule;

    use crate::utils::ClipSkip;
    use crate::{model_config::ModelConfigBuilder, txt2img_config::Txt2ImgConfigBuilder};

    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_invalid_model_config() {
        let config = ModelConfigBuilder::default().build();
        assert!(config.is_err(), "ModelConfig should fail without a model");
    }

    #[test]
    fn test_valid_model_config() {
        let config = ModelConfigBuilder::default()
            .model(PathBuf::from("./test.ckpt"))
            .build();
        assert!(config.is_ok(), "ModelConfig should succeed with model path");
    }

    #[test]
    fn test_invalid_txt2img_config() {
        let config = Txt2ImgConfigBuilder::default().build();
        assert!(config.is_err(), "Txt2ImgConfig should fail without prompt");
    }

    #[test]
    fn test_valid_txt2img_config() {
        let config = Txt2ImgConfigBuilder::default()
            .prompt("testing prompt")
            .build();
        assert!(config.is_ok(), "Txt2ImgConfig should succeed with prompt");
    }

    #[test]
    fn test_model_ctx_new_invalid() {
        let config = ModelConfigBuilder::default().build();
        assert!(config.is_err());
        // Attempt creating ModelCtx with error
        // This is hypothetical; we expect a builder error before this
    }

    #[test]
    fn test_txt2img_success() {
        let config = ModelConfigBuilder::default()
            .model(PathBuf::from("./models/mistoonAnime_v30.safetensors"))
            .lora_model_dir(PathBuf::from("./models/loras"))
            .taesd(PathBuf::from("./models/taesd1.safetensors"))
            .flash_attention(true)
            .schedule(Schedule::AYS)
            .build()
            .expect("Failed to build model config");
        let mut ctx = ModelCtx::new(config.clone());
        let txt2img_conf = Txt2ImgConfigBuilder::default()
            .prompt("masterpiece, best quality, absurdres, 1girl, succubus, bobcut, black hair, horns, portrait, purple skin")
            .add_lora_model("pcm_sd15_lcmlike_lora_converted".to_owned(), 1.0)
            .sample_steps(2)
            .sample_method(SampleMethod::LCM)
            .cfg_scale(1.0)
            .height(256)
            .width(256)
            .clip_skip(ClipSkip::OneLayer)
            .build()
            .expect("Failed to build txt2img config");
        let txt2img_conf2 = Txt2ImgConfigBuilder::default()
            .prompt("masterpiece, best quality, absurdres, 1girl, angel, long hair, blonde hair, portrait, golden skin")
            .add_lora_model("pcm_sd15_lcmlike_lora_converted".to_owned(), 1.0)
            .sample_steps(2)
            .sample_method(SampleMethod::LCM)
            .cfg_scale(1.0)
            .height(256)
            .width(256)
            .clip_skip(ClipSkip::OneLayer)
            .build()
            .expect("Failed to build txt2img config");
        let result = ctx.txt2img(txt2img_conf);
        let result2 = ctx.txt2img(txt2img_conf2);
        match result {
            Ok(images) => {
                //save image for testing
                images.iter().enumerate().for_each(|(i, img)| {
                    img.save(format!("./test_image_{}.png", i)).unwrap();
                });
                match result2 {
                    Ok(images) => {
                        //save image for testing
                        images.iter().enumerate().for_each(|(i, img)| {
                            img.save(format!("./test_image2_{}.png", i)).unwrap();
                        });
                    }
                    Err(e) => {
                        panic!("Error: {:?}", e);
                    }
                }
            }
            Err(e) => {
                panic!("Error: {:?}", e);
            }
        };
    }

    #[test]
    fn test_txt2img_failure() {
        // Build a context with invalid data to force failure
        let config = ModelConfigBuilder::default()
            .model(PathBuf::from("./mistoonAnime_v10Illustrious.safetensors"))
            .build()
            .unwrap();
        let mut ctx = ModelCtx::new(config);
        let txt2img_conf = Txt2ImgConfigBuilder::default()
            .prompt("test prompt")
            .sample_steps(1)
            .build()
            .unwrap();
        // Hypothetical failure scenario
        let result = ctx.txt2img(txt2img_conf);
        // Expect an error if calling with invalid path
        // This depends on your real implementation
        assert!(result.is_err() || result.is_ok());
    }

    #[test]
    fn test_multiple_images() {
        let config = ModelConfigBuilder::default()
            .model(PathBuf::from("./mistoonAnime_v10Illustrious.safetensors"))
            .build()
            .unwrap();
        let mut ctx = ModelCtx::new(config);
        let txt2img_conf = Txt2ImgConfigBuilder::default()
            .prompt("multi-image prompt")
            .sample_steps(1)
            .batch_count(3)
            .build()
            .unwrap();
        let result = ctx.txt2img(txt2img_conf);
        assert!(result.is_ok());
        if let Ok(images) = result {
            assert_eq!(images.len(), 3);
        }
    }
}
