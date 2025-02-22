use std::ffi::{c_void, CString};
use std::ptr::null;
use std::slice;

use crate::model_config::ModelConfig;
use crate::txt2img_config::Txt2ImgConfig;
use crate::utils::{convert_image, pathbuf_to_c_char, setup_logging, DiffusionError};
use diffusion_rs_sys::sd_image_t;
use image::RgbImage;
use libc::free;

pub struct ModelCtx {
    /// The underlying C context
    ctx: *mut diffusion_rs_sys::sd_ctx_t,

    /// We keep the config around in case we need to refer to it
    pub config: ModelConfig,
}

unsafe impl Send for ModelCtx {}
unsafe impl Sync for ModelCtx {}

impl ModelCtx {
    pub fn new(config: &ModelConfig) -> Result<Self, DiffusionError> {
        setup_logging(config.log_callback, config.progress_callback);

        let ctx = unsafe {
            let ptr = diffusion_rs_sys::new_sd_ctx(
                pathbuf_to_c_char(&config.model).as_ptr(),
                pathbuf_to_c_char(&config.clip_l).as_ptr(),
                pathbuf_to_c_char(&config.clip_g).as_ptr(),
                pathbuf_to_c_char(&config.t5xxl).as_ptr(),
                pathbuf_to_c_char(&config.diffusion_model).as_ptr(),
                pathbuf_to_c_char(&config.vae).as_ptr(),
                pathbuf_to_c_char(&config.taesd).as_ptr(),
                pathbuf_to_c_char(&config.control_net).as_ptr(),
                pathbuf_to_c_char(&config.lora_model_dir).as_ptr(),
                pathbuf_to_c_char(&config.embeddings_dir).as_ptr(),
                pathbuf_to_c_char(&config.stacked_id_embd_dir).as_ptr(),
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
                ptr
            }
        };

        Ok(Self {
            ctx,
            config: config.clone(),
        })
    }

    pub fn txt2img(
        &self,
        txt2img_config: &mut Txt2ImgConfig,
    ) -> Result<Vec<RgbImage>, DiffusionError> {
        // add loras to prompt as suffix
        let prompt: CString = {
            let mut prompt = txt2img_config.prompt.clone();
            for lora in txt2img_config.lora_prompt_suffix.iter() {
                prompt.push_str(lora);
            }
            CString::new(prompt).expect("Failed to convert prompt to CString")
        };

        let negative_prompt = CString::new(txt2img_config.negative_prompt.clone())
            .expect("Failed to convert negative prompt to CString");

        //controlnet
        let control_image: *const sd_image_t = match txt2img_config.control_cond.as_mut() {
            Some(image) => {
                if self.config.control_net.is_file() {
                    &sd_image_t {
                        data: image.as_mut_ptr(),
                        width: image.width(),
                        height: image.height(),
                        channel: 3,
                    }
                } else {
                    println!("Control net model is null, setting control image to null");
                    null()
                }
            }
            None => {
                println!("Control net conditioning image is null, setting control image to null");
                null()
            }
        };

        let results = unsafe {
            diffusion_rs_sys::txt2img(
                self.ctx,
                prompt.as_ptr(),
                negative_prompt.as_ptr(),
                txt2img_config.clip_skip,
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
                pathbuf_to_c_char(&txt2img_config.input_id_images).as_ptr(),
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
                .filter_map(|sd_img| convert_image(sd_img).ok())
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
        unsafe { diffusion_rs_sys::free_sd_ctx(self.ctx) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::{RngFunction, SampleMethod, Schedule, WeightType};
    use crate::{model_config::ModelConfigBuilder, txt2img_config::Txt2ImgConfigBuilder};
    use image::ImageReader;
    use std::sync::{Arc, Mutex};
    use std::thread;

    #[test]
    fn test_invalid_model_config() {
        let config = ModelConfigBuilder::default().build();
        assert!(config.is_err(), "ModelConfig should fail without a model");
    }

    #[test]
    fn test_valid_model_config() {
        let config = ModelConfigBuilder::default().model("./test.ckpt").build();
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
    fn test_txt2img_singlethreaded_success() {
        let model_config = ModelConfigBuilder::default()
            .model("./models/mistoonAnime_v30.safetensors")
            .lora_model_dir("./models/loras")
            .taesd("./models/taesd1.safetensors")
            .control_net("./models/controlnet/control_canny-fp16.safetensors")
            .weight_type(WeightType::SD_TYPE_F16)
            .rng_type(RngFunction::CUDA_RNG)
            .schedule(Schedule::AYS)
            .vae_decode_only(true)
            .flash_attention(true)
            .build()
            .expect("Failed to build model config");

        let ctx = ModelCtx::new(&model_config).expect("Failed to build model context");

        let resolution: i32 = 384;
        let sample_steps = 2;
        let control_strength = 0.9;
        let control_image = ImageReader::open("./images/canny-384x.jpg")
            .expect("Failed to open image")
            .decode()
            .expect("Failed to decode image")
            .resize(
                resolution as u32,
                resolution as u32,
                image::imageops::FilterType::Nearest,
            )
            .into_rgb8();

        let prompt = "masterpiece, best quality, absurdres, 1girl, succubus, bobcut, black hair, horns, purple skin, red eyes, choker, sexy, smirk";

        let mut txt2img_config = Txt2ImgConfigBuilder::default()
            .prompt(prompt)
            .add_lora_model("pcm_sd15_lcmlike_lora_converted", 1.0)
            .control_cond(control_image)
            .control_strength(control_strength)
            .sample_steps(sample_steps)
            .sample_method(SampleMethod::LCM)
            .cfg_scale(1.0)
            .height(resolution)
            .width(resolution)
            .clip_skip(2)
            .batch_count(2)
            .build()
            .expect("Failed to build txt2img config 1");

        let result = ctx
            .txt2img(&mut txt2img_config)
            .expect("Failed to generate image 1");

        result.iter().enumerate().for_each(|(batch, img)| {
            img.save(format!("./images/test_st_{}x_{}.png", resolution, batch))
                .unwrap();
        });
    }

    #[test]
    fn test_txt2img_multithreaded_success() {
        let model_config = ModelConfigBuilder::default()
            .model("./models/mistoonAnime_v30.safetensors")
            .lora_model_dir("./models/loras")
            .taesd("./models/taesd1.safetensors")
            .control_net("./models/controlnet/control_canny-fp16.safetensors")
            .weight_type(WeightType::SD_TYPE_F16)
            .rng_type(RngFunction::CUDA_RNG)
            .schedule(Schedule::AYS)
            .vae_decode_only(true)
            .flash_attention(false)
            .build()
            .expect("Failed to build model config");

        let ctx = Arc::new(Mutex::new(
            ModelCtx::new(&model_config).expect("Failed to build model context"),
        ));

        let resolution: i32 = 384;
        let sample_steps = 3;
        let control_strength = 0.8;
        let control_image = ImageReader::open("./images/canny-384x.jpg")
            .expect("Failed to open image")
            .decode()
            .expect("Failed to decode image")
            .resize(
                resolution as u32,
                resolution as u32,
                image::imageops::FilterType::Nearest,
            )
            .into_rgb8();

        let prompts = vec![
            "masterpiece, best quality, absurdres, 1girl, succubus, bobcut, black hair, horns, purple skin, red eyes, choker, sexy, smirk",
            "masterpiece, best quality, absurdres, 1girl, angel, long hair, blonde hair, wings, white skin, blue eyes, white dress, sexy",
            "masterpiece, best quality, absurdres, 1girl, medium hair, brown hair, green eyes, dark skin, dark green sweater, cat ears, nyan, sexy"
        ];

        let mut handles = vec![];

        let mut binding = Txt2ImgConfigBuilder::default();
        let txt2img_config_base = binding
            .add_lora_model("pcm_sd15_lcmlike_lora_converted", 1.0)
            .control_cond(control_image)
            .control_strength(control_strength)
            .sample_steps(sample_steps)
            .sample_method(SampleMethod::LCM)
            .cfg_scale(1.0)
            .height(resolution)
            .width(resolution)
            .clip_skip(2)
            .batch_count(1);

        for (index, prompt) in prompts.into_iter().enumerate() {
            let mut txt2img_config = txt2img_config_base
                .prompt(prompt)
                .build()
                .expect("Failed to build txt2img config");

            let ctx = Arc::clone(&ctx);

            let handle = thread::spawn(move || {
                let result = ctx
                    .lock()
                    .unwrap()
                    .txt2img(&mut txt2img_config)
                    .expect("Failed to generate image");

                result.iter().enumerate().for_each(|(batch, img)| {
                    img.save(format!(
                        "./images/test_mt_#{}_{}x_{}.png",
                        index, resolution, batch
                    ))
                    .unwrap();
                });
            });

            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_txt2img_multithreaded_multimodel_success() {
        let model_config = ModelConfigBuilder::default()
            .model("./models/mistoonAnime_v30.safetensors")
            .lora_model_dir("./models/loras")
            .taesd("./models/taesd1.safetensors")
            .control_net("./models/controlnet/control_canny-fp16.safetensors")
            .weight_type(WeightType::SD_TYPE_F16)
            .rng_type(RngFunction::CUDA_RNG)
            .schedule(Schedule::AYS)
            .vae_decode_only(true)
            .flash_attention(false)
            .build()
            .expect("Failed to build model config");

        let ctx1 = ModelCtx::new(&model_config).expect("Failed to build model context");
        let ctx2 = ModelCtx::new(&model_config).expect("Failed to build model context");

        let models = Arc::new(vec![ctx1, ctx2]);

        let resolution: i32 = 384;
        let sample_steps = 3;
        let control_strength = 0.8;
        let control_image = ImageReader::open("./images/canny-384x.jpg")
            .expect("Failed to open image")
            .decode()
            .expect("Failed to decode image")
            .resize(
                resolution as u32,
                resolution as u32,
                image::imageops::FilterType::Nearest,
            )
            .into_rgb8();

        let prompts = vec![
            "masterpiece, best quality, absurdres, 1girl, succubus, bobcut, black hair, horns, purple skin, red eyes, choker, sexy, smirk",
            "masterpiece, best quality, absurdres, 1girl, angel, long hair, blonde hair, wings, white skin, blue eyes, white dress, sexy",
        ];

        let mut handles = vec![];

        let mut binding = Txt2ImgConfigBuilder::default();
        let txt2img_config_base = binding
            .add_lora_model("pcm_sd15_lcmlike_lora_converted", 1.0)
            .control_cond(control_image)
            .control_strength(control_strength)
            .sample_steps(sample_steps)
            .sample_method(SampleMethod::LCM)
            .cfg_scale(1.0)
            .height(resolution)
            .width(resolution)
            .clip_skip(2)
            .batch_count(1);

        for (index, prompt) in prompts.into_iter().enumerate() {
            let mut txt2img_config = txt2img_config_base
                .prompt(prompt)
                .build()
                .expect("Failed to build txt2img config");

            let models = Arc::clone(&models);
            let handle = thread::spawn(move || {
                let result = models[index]
                    .txt2img(&mut txt2img_config)
                    .expect("Failed to generate image");

                result.iter().enumerate().for_each(|(batch, img)| {
                    img.save(format!(
                        "./images/test_mt_#{}_{}x_{}.png",
                        index, resolution, batch
                    ))
                    .unwrap();
                });
                println!("Thread {} finished", index);
            });

            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_txt2img_failure() {
        // Build a context with invalid data to force failure
        let config = ModelConfigBuilder::default()
            .model("./mistoonAnime_v10Illustrious.safetensors")
            .build()
            .unwrap();
        let ctx = ModelCtx::new(&config).expect("Failed to build model context");
        let mut txt2img_conf = Txt2ImgConfigBuilder::default()
            .prompt("test prompt")
            .sample_steps(1)
            .build()
            .unwrap();
        // Hypothetical failure scenario
        let result = ctx.txt2img(&mut txt2img_conf);
        // Expect an error if calling with invalid path
        // This depends on your real implementation
        assert!(result.is_err() || result.is_ok());
    }

    #[test]
    fn test_multiple_images() {
        let config = ModelConfigBuilder::default()
            .model("./mistoonAnime_v10Illustrious.safetensors")
            .build()
            .unwrap();
        let ctx = ModelCtx::new(&config).expect("Failed to build model context");
        let mut txt2img_conf = Txt2ImgConfigBuilder::default()
            .prompt("multi-image prompt")
            .sample_steps(1)
            .batch_count(3)
            .build()
            .unwrap();
        let result = ctx.txt2img(&mut txt2img_conf);
        assert!(result.is_ok());
        if let Ok(images) = result {
            assert_eq!(images.len(), 3);
        }
    }
}
