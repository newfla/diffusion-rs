use std::ffi::{CString, c_void};
use std::mem::ManuallyDrop;
use std::ptr::null;
use std::slice;

use crate::model_config::ModelConfig;
use crate::txt2img_config::Txt2ImgConfig;
use crate::types::{DiffusionError, LogCallback, ProgressCallback, SdLogLevel};
use crate::utils::{convert_image, pathbuf_to_c_char};

use diffusion_rs_sys::sd_image_t;
use image::RgbImage;
use libc::free;

#[derive(Debug)]
pub struct ModelCtx {
    /// The underlying C context
    ctx: *mut diffusion_rs_sys::sd_ctx_t,

    /// We keep the config around in case we need to refer to it
    pub config: ModelConfig,
}

unsafe impl Send for ModelCtx {}
// unsafe impl Sync for ModelCtx {}

impl ModelCtx {
    pub fn new(config: &ModelConfig) -> Result<Self, DiffusionError> {
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

    pub fn set_log_callback<F>(on_log: F) -> ()
    where
        F: Fn(SdLogLevel, String) + Send + Sync + 'static,
    {
        // Create a new log callback
        let t = ManuallyDrop::new(LogCallback::new(on_log));
        unsafe { diffusion_rs_sys::sd_set_log_callback(t.callback(), t.user_data()) };
    }

    pub fn set_progress_callback<F>(on_progress: F) -> ()
    where
        F: Fn(i32, i32, f32) + Send + Sync + 'static,
    {
        // Create a new progress callback
        let t = ManuallyDrop::new(ProgressCallback::new(on_progress));
        unsafe { diffusion_rs_sys::sd_set_progress_callback(t.callback(), t.user_data()) };
    }

    pub fn txt2img(&self, txt2img_config: &Txt2ImgConfig) -> Result<Vec<RgbImage>, DiffusionError> {
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
        let control_image: *const sd_image_t = match txt2img_config.control_cond.as_ref() {
            Some(image) => {
                if self.config.control_net.is_file() {
                    &sd_image_t {
                        data: image.as_ptr().cast_mut(),
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
                txt2img_config.eta,
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
                txt2img_config.skip_layer.clone().as_mut_ptr(),
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
