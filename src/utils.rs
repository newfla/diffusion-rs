use diffusion_rs_sys::sd_get_system_info;
use diffusion_rs_sys::sd_log_level_t;
use diffusion_rs_sys::sd_set_log_callback;
use image::ImageBuffer;
use image::Rgb;
use image::RgbImage;
use libc::free;
use std::ffi::c_char;
use std::ffi::c_void;
use std::ffi::CStr;
use std::ffi::CString;

use std::path::Path;
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
    // #[error("Free Params Immediately is set to true, which means that the params are freed after forward. This means that the model can only be used once")]
    // FreeParamsImmediately,
}

#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum SDImageError {
    #[error("Failed to convert image buffer to Rust type")]
    AllocationError,
    #[error("The image buffer has a different length than expected")]
    DifferentLength,
}

#[repr(i32)]
#[non_exhaustive]
#[derive(Debug, Default, Clone, Hash, PartialEq, Eq)]
/// Ignore the lower X layers of CLIP network
pub enum ClipSkip {
    /// Will be [ClipSkip::None] for SD1.x, [ClipSkip::OneLayer] for SD2.x
    #[default]
    Unspecified = 0,
    None = 1,
    OneLayer = 2,
}

#[derive(Debug, Clone, Default)]
pub struct CLibString(pub CString);

impl CLibString {
    pub fn as_ptr(&self) -> *const c_char {
        self.0.as_ptr()
    }
}

impl From<&str> for CLibString {
    fn from(value: &str) -> Self {
        Self(CString::new(value).unwrap())
    }
}

impl From<String> for CLibString {
    fn from(value: String) -> Self {
        Self(CString::new(value).unwrap())
    }
}

#[derive(Debug, Clone, Default)]
pub struct CLibPath(CString);

impl CLibPath {
    pub fn as_ptr(&self) -> *const c_char {
        self.0.as_ptr()
    }
}

impl From<PathBuf> for CLibPath {
    fn from(value: PathBuf) -> Self {
        Self(CString::new(value.to_str().unwrap_or_default()).unwrap())
    }
}

impl From<&Path> for CLibPath {
    fn from(value: &Path) -> Self {
        Self(CString::new(value.to_str().unwrap_or_default()).unwrap())
    }
}

/// Specify the range function
pub use diffusion_rs_sys::rng_type_t as RngFunction;

/// Denoiser sigma schedule
pub use diffusion_rs_sys::schedule_t as Schedule;

/// Weight type
pub use diffusion_rs_sys::sd_type_t as WeightType;

/// Sampling methods
pub use diffusion_rs_sys::sample_method_t as SampleMethod;

use diffusion_rs_sys::sd_image_t;

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

extern "C" fn my_log_callback(level: sd_log_level_t, text: *const c_char, _data: *mut c_void) {
    unsafe {
        // Convert C string to Rust &str and print it.
        if !text.is_null() {
            let msg = CStr::from_ptr(text).to_str().unwrap_or("Invalid UTF-8");
            print!("({:?}): {}", level, msg);
        }
    }
}

pub fn setup_logging() {
    unsafe {
        sd_set_log_callback(Some(my_log_callback), std::ptr::null_mut());
    }
}
