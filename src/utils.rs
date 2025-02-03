use image::ImageBuffer;
use image::Rgb;
use image::RgbImage;
use std::ffi::c_char;
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
    #[error("sd_ctx_t is None")]
    NoContext,
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
#[derive(Debug, Default, Copy, Clone, Hash, PartialEq, Eq)]
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

/// Image buffer Type
pub use diffusion_rs_sys::sd_image_t;

#[derive(Debug, Clone)]
pub struct SdImageContainer {
    // Wrap the raw external type.
    inner: sd_image_t,
}

impl SdImageContainer {
    pub fn as_ptr(&self) -> *const sd_image_t {
        &self.inner
    }
}

impl From<sd_image_t> for SdImageContainer {
    fn from(inner: sd_image_t) -> Self {
        Self { inner }
    }
}

impl TryFrom<RgbImage> for SdImageContainer {
    type Error = SDImageError;

    fn try_from(img: RgbImage) -> Result<Self, Self::Error> {
        let (width, height) = img.dimensions();
        // For an RGB image, we have 3 channels.
        let channel = 3u32;
        let expected_len = (width * height * channel) as usize;

        // Convert the image into its raw pixel data (a Vec<u8>).
        let pixel_data: Vec<u8> = img.into_raw();

        // Ensure that the pixel data is of the expected length.
        if pixel_data.len() != expected_len {
            return Err(SDImageError::DifferentLength);
        }

        let data_ptr = unsafe {
            let ptr = libc::malloc(expected_len) as *mut u8;
            if ptr.is_null() {
                return Err(SDImageError::AllocationError);
            }
            std::ptr::copy_nonoverlapping(pixel_data.as_ptr(), ptr, expected_len);
            ptr
        };

        Ok(SdImageContainer {
            inner: sd_image_t {
                width,
                height,
                channel,
                data: data_ptr,
            },
        })
    }
}

impl Drop for SdImageContainer {
    fn drop(&mut self) {
        unsafe {
            libc::free(self.inner.data as *mut libc::c_void);
        }
    }
}

impl TryFrom<SdImageContainer> for RgbImage {
    type Error = SDImageError;

    fn try_from(sd_image: SdImageContainer) -> Result<Self, Self::Error> {
        let len = (sd_image.inner.width * sd_image.inner.height * sd_image.inner.channel) as usize;
        let raw_pixels = unsafe { slice::from_raw_parts(sd_image.inner.data, len) };
        let buffer = raw_pixels.to_vec();
        let buffer = ImageBuffer::<Rgb<u8>, _>::from_raw(
            sd_image.inner.width as u32,
            sd_image.inner.height as u32,
            buffer,
        );
        Ok(match buffer {
            Some(buffer) => RgbImage::from(buffer),
            None => return Err(SDImageError::AllocationError),
        })
    }
}
