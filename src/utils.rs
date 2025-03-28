use image::ImageBuffer;
use image::Rgb;
use image::RgbImage;
use std::ffi::CString;
use std::path::PathBuf;
use std::slice;

use diffusion_rs_sys::sd_image_t;

use crate::types::SDImageError;

pub fn pathbuf_to_c_char(path: &PathBuf) -> CString {
    let path_str = path
        .to_str()
        .expect("PathBuf contained non-UTF-8 characters");
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
