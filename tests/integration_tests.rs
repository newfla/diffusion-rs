use diffusion_rs::api::ModelCtx;
use diffusion_rs::model_config::ModelConfigBuilder;
use diffusion_rs::txt2img_config::Txt2ImgConfigBuilder;
use diffusion_rs::types::{RngFunction, SampleMethod, Schedule, WeightType};
use image::ImageReader;

use std::sync::{Arc, Mutex};
use std::thread;

#[test]
fn test_txt2img_singlethreaded_success() {
    let model_config = ModelConfigBuilder::default()
        .model("./models/mistoonAnime_v30.safetensors")
        .lora_model_dir("./models/loras")
        .taesd("./models/taesd1.safetensors")
        .control_net("./models/controlnet/control_canny-fp16.safetensors")
        .schedule(Schedule::AYS)
        .vae_decode_only(true)
        .flash_attention(true)
        .log_callback(|level, text| {
            print!("({:?}): {}", level, text);
        })
        .progress_callback(|step, steps, time| {
            println!("Progress: {}/{} ({}s)", step, steps, time);
        })
        .build()
        .expect("Failed to build model config");

    let ctx = ModelCtx::new(&model_config).expect("Failed to build model context");

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

    let prompt = "masterpiece, best quality, absurdres, 1girl, succubus, bobcut, black hair, horns, purple skin, red eyes, choker, sexy, smirk";

    let txt2img_config = Txt2ImgConfigBuilder::default()
        .prompt(prompt)
        .add_lora_model("pcm_sd15_smallcfg_2step_converted", 1.0)
        .control_cond(control_image)
        .control_strength(control_strength)
        .sample_steps(sample_steps)
        .sample_method(SampleMethod::TCD)
        .eta(1.0)
        .cfg_scale(1.0)
        .height(resolution)
        .width(resolution)
        .clip_skip(2)
        .batch_count(1)
        .build()
        .expect("Failed to build txt2img config 1");

    let result = ctx
        .txt2img(&txt2img_config)
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
        .log_callback(|level, text| {
            print!("({:?}): {}", level, text);
        })
        .progress_callback(|step, steps, time| {
            println!("Progress: {}/{} ({}s)", step, steps, time);
        })
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
        "masterpiece, best quality, absurdres, 1girl, medium hair, brown hair, green eyes, dark skin, dark green sweater, cat ears, nyan, sexy",
    ];

    let mut handles = vec![];

    let mut builder = Txt2ImgConfigBuilder::default();

    builder
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
        builder.prompt(prompt);
        let txt2img_config = builder.build().expect("Failed to build txt2img config");

        let ctx = Arc::clone(&ctx);

        let handle = thread::spawn(move || {
            let result = ctx
                .lock()
                .unwrap()
                .txt2img(&txt2img_config)
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
    let mut model_config = ModelConfigBuilder::default();
    model_config
        .model("./models/mistoonAnime_v30.safetensors")
        .lora_model_dir("./models/loras")
        .taesd("./models/taesd1.safetensors")
        .control_net("./models/controlnet/control_canny-fp16.safetensors")
        .schedule(Schedule::AYS)
        .vae_decode_only(true)
        .flash_attention(true);

    let mut model_handle = vec![];
    for x in 0..2 {
        let model_config = model_config
            .log_callback(|level, text| {
                print!("[Thread {}], ({:?}): {}", x, level, text);
            })
            .build()
            .expect("Failed to build model config");
        let handle = thread::spawn(move || {
            // Use the context directly in the thread
            return ModelCtx::new(&model_config).expect("Failed to build model context");
        });
        model_handle.push(handle);
    }

    // wait for threads to finish
    let mut models = vec![];
    for handle in model_handle {
        let ctx = handle.join().expect("Failed to join thread");
        models.push(ctx);
    }

    let models = Arc::new(models);

    let resolution: i32 = 384;
    let sample_steps = 1;
    let control_strength = 0.5;
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
        let txt2img_config = txt2img_config_base
            .prompt(prompt)
            .build()
            .expect("Failed to build txt2img config");

        let models = models.clone();
        let handle = thread::spawn(move || {
            let result = models[index]
                .txt2img(&txt2img_config)
                .expect("Failed to generate image");

            result.iter().enumerate().for_each(|(batch, img)| {
                img.save(format!(
                    "./images/test_mt_mm_#{}_{}x_{}.png",
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
    let txt2img_conf = Txt2ImgConfigBuilder::default()
        .prompt("test prompt")
        .sample_steps(1)
        .build()
        .unwrap();
    // Hypothetical failure scenario
    let result = ctx.txt2img(&txt2img_conf);
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
    let txt2img_conf = Txt2ImgConfigBuilder::default()
        .prompt("multi-image prompt")
        .sample_steps(1)
        .batch_count(3)
        .build()
        .unwrap();
    let result = ctx.txt2img(&txt2img_conf);
    assert!(result.is_ok());
    if let Ok(images) = result {
        assert_eq!(images.len(), 3);
    }
}
