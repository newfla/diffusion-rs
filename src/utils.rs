use diffusion_rs_sys::sd_log_level_t;
use diffusion_rs_sys::sd_set_log_callback;
use diffusion_rs_sys::sd_set_progress_callback;
use image::ImageBuffer;
use image::Rgb;
use image::RgbImage;
use std::ffi::CStr;
use std::ffi::CString;
use std::ffi::c_char;
use std::ffi::c_void;
use std::path::PathBuf;
use std::slice;
use thiserror::Error;

#[non_exhaustive]
#[derive(Error, Debug)]
/// Error that can occurs while forwarding models
pub enum DiffusionError {
    #[error("The underling stablediffusion.cpp function returned NULL")]
    Forward,
    #[error("The underling stbi_write_image function returned 0 while saving image {0}/{1})")]
    StoreImages(usize, i32),
    #[error("The underling upscaler model returned a NULL image")]
    Upscaler,
    #[error("raw_ctx is None")]
    NoContext,
    #[error("new_sd_ctx returned null")]
    NewContextFailure,
    #[error("SD image conversion error: {0}")]
    SDImageError(#[from] SDImageError),
}

#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum SDImageError {
    #[error("Failed to convert image buffer to Rust type")]
    AllocationError,
    #[error("The image buffer has a different length than expected")]
    DifferentLength,
}

/// Specify the range function
pub use diffusion_rs_sys::rng_type_t as RngFunction;

/// Denoiser sigma schedule
pub use diffusion_rs_sys::schedule_t as Schedule;

/// Weight type
pub use diffusion_rs_sys::sd_type_t as WeightType;

/// Sampling methods
pub use diffusion_rs_sys::sample_method_t as SampleMethod;

//log level
pub use diffusion_rs_sys::sd_log_level_t as SdLogLevel;

use diffusion_rs_sys::sd_image_t;

pub fn pathbuf_to_c_char(path: &PathBuf) -> CString {
    let path_str = path
        .to_str()
        .expect("PathBuf contained non-UTF-8 characters");
    // Create a CString which adds a null terminator.
    CString::new(path_str).expect("CString conversion failed")
}

pub fn convert_image(sd_image: &sd_image_t) -> Result<RgbImage, SDImageError> {
    let len = (sd_image.width * sd_image.height * sd_image.channel) as usize;
    let raw_pixels = unsafe { slice::from_raw_parts(sd_image.data, len) };
    let buffer = raw_pixels.to_vec();
    let buffer =
        ImageBuffer::<Rgb<u8>, _>::from_raw(sd_image.width as u32, sd_image.height as u32, buffer);
    Ok(match buffer {
        Some(buffer) => RgbImage::from(buffer),
        None => return Err(SDImageError::AllocationError),
    })
}

extern "C" fn default_log_callback(level: sd_log_level_t, text: *const c_char, _data: *mut c_void) {
    unsafe {
        // Convert C string to Rust &str and print it.
        if !text.is_null() {
            let msg = CStr::from_ptr(text)
                .to_str()
                .unwrap_or("LOG ERROR: Invalid UTF-8");
            print!("({:?}): {}", level, msg);
        }
    }
}

type UnsafeLogCallbackFn =
    extern "C" fn(level: sd_log_level_t, text: *const c_char, data: *mut c_void);

type UnsafeProgressCallbackFn = extern "C" fn(step: i32, steps: i32, time: f32, data: *mut c_void);

pub fn setup_logging(
    log_callback: Option<UnsafeLogCallbackFn>,
    progress_callback: Option<UnsafeProgressCallbackFn>,
    logging_data: *mut c_void,
) {
    unsafe {
        match log_callback {
            Some(callback) => sd_set_log_callback(Some(callback), logging_data),
            None => sd_set_log_callback(Some(default_log_callback), logging_data),
        };
        match progress_callback {
            Some(callback) => sd_set_progress_callback(Some(callback), logging_data),
            None => (),
        };
    }
}

#[derive(Debug, Clone)]
pub struct Data {
    pub data: *mut c_void,
}

unsafe impl Send for Data {}
