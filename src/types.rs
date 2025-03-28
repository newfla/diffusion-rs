/// Specify the range function
pub use diffusion_rs_sys::rng_type_t as RngFunction;

/// Denoiser sigma schedule
pub use diffusion_rs_sys::schedule_t as Schedule;

/// Weight type
pub use diffusion_rs_sys::sd_type_t as WeightType;

/// Sampling methods
pub use diffusion_rs_sys::sample_method_t as SampleMethod;

/// Log Level
pub use diffusion_rs_sys::sd_log_level_t as SdLogLevel;

#[non_exhaustive]
#[derive(thiserror::Error, Debug)]
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
