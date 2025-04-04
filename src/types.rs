use std::ffi::CStr;
use std::ffi::c_char;
use std::ffi::c_void;

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

#[derive(Debug)]
pub struct LogCallback {
    callback: Option<UnsafeLogCallbackFn>,
    free_user_data: unsafe fn(*mut c_void),
    user_data: *mut c_void,
}

impl LogCallback {
    pub fn new<F: Fn(SdLogLevel, String) + Send + Sync + 'static>(f: F) -> Self {
        unsafe extern "C" fn callback<F: Fn(SdLogLevel, String) + Send + Sync + 'static>(
            level: SdLogLevel,
            text: *const c_char,
            data: *mut c_void,
        ) {
            let f: &F = unsafe { &*data.cast_const().cast::<F>() };
            // convert input and pass to closure
            let msg = unsafe { CStr::from_ptr(text) }
                .to_str()
                .unwrap_or("LOG ERROR: Invalid UTF-8");
            f(level, msg.to_string());
        }

        unsafe fn free_user_data<F>(user_data: *mut c_void) {
            let user_data = user_data.cast::<F>();
            unsafe { _ = Box::from_raw(user_data) }
        }

        let user_data = Box::into_raw(Box::new(f));

        Self {
            callback: Some(callback::<F>),
            free_user_data: free_user_data::<F>,
            user_data: user_data.cast(),
        }
    }

    pub fn user_data(&self) -> *mut c_void {
        self.user_data
    }

    pub fn callback(&self) -> Option<UnsafeLogCallbackFn> {
        self.callback
    }
}

// SAFETY: A Callback can only be constructed with
// LogCallback::new, which requires F to be Send and Sync.
unsafe impl Send for LogCallback {}
unsafe impl Sync for LogCallback {}

impl Drop for LogCallback {
    fn drop(&mut self) {
        unsafe { (self.free_user_data)(self.user_data) }
    }
}

#[derive(Debug)]
pub struct ProgressCallback {
    callback: Option<UnsafeProgressCallbackFn>,
    free_user_data: unsafe fn(*mut c_void),
    user_data: *mut c_void,
}

impl ProgressCallback {
    pub fn new<F: Fn(i32, i32, f32) + Send + Sync + 'static>(f: F) -> Self {
        unsafe extern "C" fn callback<F: Fn(i32, i32, f32) + Send + Sync + 'static>(
            step: i32,
            steps: i32,
            time: f32,
            data: *mut c_void,
        ) {
            let f: &F = unsafe { &*data.cast_const().cast::<F>() };
            // convert input and pass to closure
            f(step, steps, time);
        }

        unsafe fn free_user_data<F>(user_data: *mut c_void) {
            let user_data = user_data.cast::<F>();
            unsafe { _ = Box::from_raw(user_data) }
        }

        let user_data = Box::into_raw(Box::new(f));

        Self {
            callback: Some(callback::<F>),
            free_user_data: free_user_data::<F>,
            user_data: user_data.cast(),
        }
    }

    pub fn user_data(&self) -> *mut c_void {
        self.user_data
    }

    pub fn callback(&self) -> Option<UnsafeProgressCallbackFn> {
        self.callback
    }
}

// SAFETY: A Callback can only be constructed with
// ProgressCallback::new, which requires F to be Send and Sync.
unsafe impl Send for ProgressCallback {}
unsafe impl Sync for ProgressCallback {}

impl Drop for ProgressCallback {
    fn drop(&mut self) {
        unsafe { (self.free_user_data)(self.user_data) }
    }
}

type UnsafeLogCallbackFn =
    unsafe extern "C" fn(level: SdLogLevel, text: *const c_char, data: *mut c_void);

type UnsafeProgressCallbackFn =
    unsafe extern "C" fn(step: i32, steps: i32, time: f32, data: *mut c_void);
