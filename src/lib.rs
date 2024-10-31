#![doc = include_str!("../README.md")]
/// Safer wrapper around stable-diffusion.cpp bindings
pub mod api;

/// Presets that automatically download models from <https://huggingface.co/>
pub mod preset;

/// Add additional resources to [preset::Preset]
pub mod modifier;
pub(crate) mod preset_builder;

/// Util module
pub mod util;
