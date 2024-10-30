#![doc = include_str!("../README.md")]
/// Safer wrapper around stable-diffusion.cpp bindings
pub mod api;

/// Presets that automatically downlaod models from <https://huggingface.co/>
pub mod preset;

/// Add additional resources to [crate::preset::Preset]
pub mod modifier;
pub(crate) mod preset_builder;

/// Util module
pub mod util;
