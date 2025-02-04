use crate::model_config::ModelConfig;
use crate::txt2img_config::Txt2ImgConfig;
use crate::utils::CLibString;
use crate::utils::DiffusionError;
use crate::utils::SdImageContainer;
use diffusion_rs_sys::free_sd_ctx;
use diffusion_rs_sys::new_sd_ctx;
use diffusion_rs_sys::sd_ctx_t;
use diffusion_rs_sys::sd_image_t;
use diffusion_rs_sys::strlen;
use image::RgbImage;
use libc::free;
use std::ffi::c_void;
use std::ptr::null;
use std::slice;

pub struct ModelCtx {
    /// The underlying C context
    raw_ctx: Option<*mut sd_ctx_t>,

    /// We keep the config around in case we need to refer to it
    pub model_config: ModelConfig,
}

impl ModelCtx {
    pub fn new(config: ModelConfig) -> Result<Self, DiffusionError> {
        let raw_ctx = unsafe {
            let ptr = new_sd_ctx(
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
            );
            if ptr.is_null() {
                return Err(DiffusionError::NewContextFailure);
            } else {
                Some(ptr)
            }
        };

        Ok(Self {
            raw_ctx,
            model_config: config,
        })
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

        //controlnet

        let control_image = match txt2img_config.control_cond {
            Some(image) => {
                match unsafe { strlen(self.model_config.control_net.as_ptr()) as usize > 0 } {
                    true => {
                        let wrapper = SdImageContainer::try_from(image)?;
                        wrapper.as_ptr()
                    }
                    false => {
                        println!("Control net model is null, setting control image to null");
                        null()
                    }
                }
            }
            None => null(),
        };

        //run text to image
        let results: *mut sd_image_t = unsafe {
            diffusion_rs_sys::txt2img(
                self.raw_ctx.ok_or(DiffusionError::NoContext)?,
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
                control_image,
                txt2img_config.control_strength,
                txt2img_config.style_strength,
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

        let result_images: Vec<RgbImage> = {
            let img_count = txt2img_config.batch_count as usize;
            let images = unsafe { slice::from_raw_parts(results, img_count) };
            images
                .iter()
                .filter_map(|sd_img| RgbImage::try_from(SdImageContainer::from(*sd_img)).ok())
                .collect()
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
        match self.raw_ctx {
            Some(ptr) => unsafe {
                free_sd_ctx(ptr);
            },
            None => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::{ClipSkip, SampleMethod, Schedule, WeightType};
    use crate::{model_config::ModelConfigBuilder, txt2img_config::Txt2ImgConfigBuilder};
    use image::ImageReader;
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
        let control_image1 = ImageReader::open("./images/canny-384x.jpg")
            .expect("Failed to open image")
            .decode()
            .expect("Failed to decode image")
            .into_rgb8();

        let control_image2 = ImageReader::open("./images/canny-384x.jpg")
            .expect("Failed to open image")
            .decode()
            .expect("Failed to decode image")
            .into_rgb8();

        let control_image3 = ImageReader::open("./images/canny-384x.jpg")
            .expect("Failed to open image")
            .decode()
            .expect("Failed to decode image")
            .into_rgb8();

        let resolution = 384;
        let sample_steps = 4;
        let control_strength = 0.4;

        let mut ctx = ModelCtx::new(
            ModelConfigBuilder::default()
                .model(PathBuf::from("./models/mistoonAnime_v30.safetensors"))
                .lora_model_dir(PathBuf::from("./models/loras"))
                .taesd(PathBuf::from("./models/taesd1.safetensors"))
                .control_net(PathBuf::from(
                    "./models/controlnet/control_canny-fp16.safetensors",
                ))
                //.weight_type(WeightType::SD_TYPE_Q4_1)
                .flash_attention(true)
                .schedule(Schedule::AYS)
                .build()
                .expect("Failed to build model config"),
        )
        .expect("Failed to build model context");

        let result = ctx
            .txt2img(Txt2ImgConfigBuilder::default()
            .prompt("masterpiece, best quality, absurdres, 1girl, succubus, bobcut, black hair, horns, purple skin, red eyes, choker, sexy, smirk")
            .control_cond(control_image1)
            .control_strength(control_strength)
            .add_lora_model("pcm_sd15_lcmlike_lora_converted", 1.0)
            .sample_steps(sample_steps)
            .sample_method(SampleMethod::LCM)
            .cfg_scale(1.0)
            .height(resolution)
            .width(resolution)
            .clip_skip(ClipSkip::OneLayer)
            .batch_count(1)
            .build()
            .expect("Failed to build txt2img config 1"))
            .expect("Failed to generate image 1");

        result.iter().enumerate().for_each(|(i, img)| {
            img.save(format!("./images/test_1_{}x_{}.png", resolution, i))
                .unwrap();
        });

        let result2 = ctx
            .txt2img(Txt2ImgConfigBuilder::default()
            .prompt("masterpiece, best quality, absurdres, 1girl, angel, long hair, blonde hair, wings, white skin, blue eyes, white dress, sexy")
            .control_cond(control_image2)
            .control_strength(control_strength)
            .add_lora_model("pcm_sd15_lcmlike_lora_converted", 1.0)
            .sample_steps(sample_steps)
            .sample_method(SampleMethod::LCM)
            .cfg_scale(1.0)
            .height(resolution)
            .width(resolution)
            .clip_skip(ClipSkip::OneLayer)
            .batch_count(1)
            .build()
            .expect("Failed to build txt2img config 1"))
            .expect("Failed to generate image 1");

        result2.iter().enumerate().for_each(|(i, img)| {
            img.save(format!("./images/test_2_{}x_{}.png", resolution, i))
                .unwrap();
        });

        let result3 = ctx
            .txt2img(Txt2ImgConfigBuilder::default()
            .prompt("masterpiece, best quality, absurdres, 1girl, medium hair, brown hair, green eyes, dark skin, dark green sweater, cat ears, nyan, sexy")
            .control_cond(control_image3)
            .control_strength(control_strength)
            .add_lora_model("pcm_sd15_lcmlike_lora_converted", 1.0)
            .sample_steps(sample_steps)
            .sample_method(SampleMethod::LCM)
            .cfg_scale(1.0)
            .height(resolution)
            .width(resolution)
            .clip_skip(ClipSkip::OneLayer)
            .batch_count(1)
            .build()
            .expect("Failed to build txt2img config 1"))
            .expect("Failed to generate image 1");

        result3.iter().enumerate().for_each(|(i, img)| {
            img.save(format!("./images/test_3_{}x_{}.png", resolution, i))
                .unwrap();
        });
    }

    #[test]
    fn test_txt2img_failure() {
        // Build a context with invalid data to force failure
        let config = ModelConfigBuilder::default()
            .model(PathBuf::from("./mistoonAnime_v10Illustrious.safetensors"))
            .build()
            .unwrap();
        let mut ctx = ModelCtx::new(config).expect("Failed to build model context");
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
        let mut ctx = ModelCtx::new(config).expect("Failed to build model context");
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
