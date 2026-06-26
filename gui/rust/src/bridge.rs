//! Maps GuiParams DTO to diffusion-rs builder types.
//!
//! This module bridges the FFI boundary: it takes primitive-typed fields from
//! GuiParams and converts them to the concrete Rust types that PresetBuilder,
//! ConfigBuilder and ModelConfigBuilder expect.

use std::path::PathBuf;
use std::str::FromStr;

use anyhow::{anyhow, Result};
use strum::VariantNames;

use diffusion_rs::api::{
    Config, DbCacheParamsBuilder, EasyCacheParamsBuilder, HiresParamsBuilder, ModelConfig,
    PreviewType, SpectrumCacheParamsBuilder, UCacheParamsBuilder, Upscaler,
};
use diffusion_rs::modifier::lazily_load_params_from_disk;
use diffusion_rs::preset::{
    Anima2Weight, AnimaWeight, ChromaRadianceWeight, ChromaWeight, DiffInstructStarWeight,
    ErnieImageWeight, Flux1MiniWeight, Flux1Weight, Flux2Klein4BWeight, Flux2Klein9BWeight,
    Flux2KleinBase4BWeight, Flux2KleinBase9BWeight, Flux2Weight, LongCatImageWeight,
    NitroSDRealismWeight, NitroSDVibrantWeight, OvisImageWeight, Preset, PresetBuilder,
    PresetDiscriminants, QwenImageWeight, SDXS512DreamShaperWeight, SSD1BWeight,
    TwinFlowZImageTurboExpWeight, WeightType, ZImageTurboWeight, Krea2Weight,
};
use diffusion_rs::util::set_hf_token;

use crate::gui_params::GuiParams;

/// Parse a preset discriminant string and optional weight string into a concrete
/// `Preset` enum value.
///
/// Returns a descriptive error on invalid preset or weight strings instead of
/// panicking (Pitfall 7 mitigation).
pub fn map_preset(preset_str: &str, weight_str: Option<&str>) -> Result<Preset> {
    let disc = PresetDiscriminants::from_str(preset_str)
        .map_err(|_| anyhow!("Unknown preset: '{}'. Valid presets: {:?}", preset_str, PresetDiscriminants::VARIANTS))?;

    // Helper: parse weight string to WeightType, then try_into the specific subenum.
    // If weight_str is None, use the subenum default.
    macro_rules! with_weight {
        ($variant:ident, $weight_type:ty) => {{
            let wt: $weight_type = match weight_str {
                Some(w) => {
                    let general = WeightType::from_str(w)
                        .map_err(|_| anyhow!("Unknown weight: '{}'. Valid weights for {}: {:?}", w, preset_str, <$weight_type>::VARIANTS))?;
                    general.try_into()
                        .map_err(|_| anyhow!("Weight '{}' is not valid for preset '{}'. Valid weights: {:?}", w, preset_str, <$weight_type>::VARIANTS))?
                }
                None => <$weight_type>::default(),
            };
            Ok(Preset::$variant(wt))
        }};
    }

    match disc {
        // Presets without weights
        PresetDiscriminants::StableDiffusion1_4 => Ok(Preset::StableDiffusion1_4),
        PresetDiscriminants::StableDiffusion1_5 => Ok(Preset::StableDiffusion1_5),
        PresetDiscriminants::StableDiffusion2_1 => Ok(Preset::StableDiffusion2_1),
        PresetDiscriminants::StableDiffusion3Medium => Ok(Preset::StableDiffusion3Medium),
        PresetDiscriminants::StableDiffusion3_5Medium => Ok(Preset::StableDiffusion3_5Medium),
        PresetDiscriminants::StableDiffusion3_5Large => Ok(Preset::StableDiffusion3_5Large),
        PresetDiscriminants::StableDiffusion3_5LargeTurbo => Ok(Preset::StableDiffusion3_5LargeTurbo),
        PresetDiscriminants::SDXLBase1_0 => Ok(Preset::SDXLBase1_0),
        PresetDiscriminants::SDTurbo => Ok(Preset::SDTurbo),
        PresetDiscriminants::SDXLTurbo1_0 => Ok(Preset::SDXLTurbo1_0),
        PresetDiscriminants::JuggernautXL11 => Ok(Preset::JuggernautXL11),
        PresetDiscriminants::DreamShaperXL2_1Turbo => Ok(Preset::DreamShaperXL2_1Turbo),
        PresetDiscriminants::SegmindVega => Ok(Preset::SegmindVega),
        PresetDiscriminants::HiDreamO1ImageDev => Ok(Preset::HiDreamO1ImageDev),
        PresetDiscriminants::HiDreamO1Image => Ok(Preset::HiDreamO1Image),
        PresetDiscriminants::Lens => Ok(Preset::Lens),
        PresetDiscriminants::LensTurbo => Ok(Preset::LensTurbo),
        PresetDiscriminants::BooguImage => Ok(Preset::BooguImage),
        PresetDiscriminants::BooguImageTurbo => Ok(Preset::BooguImageTurbo),


        // Presets with weights
        PresetDiscriminants::Flux1Dev => with_weight!(Flux1Dev, Flux1Weight),
        PresetDiscriminants::Flux1Schnell => with_weight!(Flux1Schnell, Flux1Weight),
        PresetDiscriminants::Flux1Mini => with_weight!(Flux1Mini, Flux1MiniWeight),
        PresetDiscriminants::Chroma => with_weight!(Chroma, ChromaWeight),
        PresetDiscriminants::NitroSDRealism => with_weight!(NitroSDRealism, NitroSDRealismWeight),
        PresetDiscriminants::NitroSDVibrant => with_weight!(NitroSDVibrant, NitroSDVibrantWeight),
        PresetDiscriminants::DiffInstructStar => with_weight!(DiffInstructStar, DiffInstructStarWeight),
        PresetDiscriminants::ChromaRadiance => with_weight!(ChromaRadiance, ChromaRadianceWeight),
        PresetDiscriminants::SSD1B => with_weight!(SSD1B, SSD1BWeight),
        PresetDiscriminants::Flux2Dev => with_weight!(Flux2Dev, Flux2Weight),
        PresetDiscriminants::ZImageTurbo => with_weight!(ZImageTurbo, ZImageTurboWeight),
        PresetDiscriminants::QwenImage => with_weight!(QwenImage, QwenImageWeight),
        PresetDiscriminants::OvisImage => with_weight!(OvisImage, OvisImageWeight),
        PresetDiscriminants::TwinFlowZImageTurboExp => with_weight!(TwinFlowZImageTurboExp, TwinFlowZImageTurboExpWeight),
        PresetDiscriminants::SDXS512DreamShaper => with_weight!(SDXS512DreamShaper, SDXS512DreamShaperWeight),
        PresetDiscriminants::Flux2Klein4B => with_weight!(Flux2Klein4B, Flux2Klein4BWeight),
        PresetDiscriminants::Flux2KleinBase4B => with_weight!(Flux2KleinBase4B, Flux2KleinBase4BWeight),
        PresetDiscriminants::Flux2Klein9B => with_weight!(Flux2Klein9B, Flux2Klein9BWeight),
        PresetDiscriminants::Flux2KleinBase9B => with_weight!(Flux2KleinBase9B, Flux2KleinBase9BWeight),
        PresetDiscriminants::Anima => with_weight!(Anima, AnimaWeight),
        PresetDiscriminants::Anima2 => with_weight!(Anima2, Anima2Weight),
        PresetDiscriminants::ErnieImage => with_weight!(ErnieImage, ErnieImageWeight),
        PresetDiscriminants::ErnieImageTurbo => with_weight!(ErnieImageTurbo, ErnieImageWeight),
        PresetDiscriminants::LongCatImage => with_weight!(LongCatImage, LongCatImageWeight),
        PresetDiscriminants::Krea2 => with_weight!(Krea2, Krea2Weight),
        PresetDiscriminants::Krea2Turbo => with_weight!(Krea2Turbo, Krea2Weight),

    }
}

/// Build `(Config, ModelConfig)` from the GUI parameters DTO.
///
/// Follows the same pattern as `cli/src/main.rs`: create a PresetBuilder, set
/// preset and prompt, then use `with_modifier` to apply optional overrides.
pub fn build_configs(params: &GuiParams) -> Result<(Config, ModelConfig)> {
    // Set HF token before building if provided
    if let Some(ref token) = params.token {
        if !token.is_empty() {
            set_hf_token(token);
        }
    }

    let preset = map_preset(&params.preset, params.weight.as_deref())?;
    let output = PathBuf::from(&params.output);
    let preview_output = PathBuf::from(&params.preview_output);

    // Clone optional params for move into closure
    let steps = params.steps;
    let width = params.width;
    let height = params.height;
    let negative_prompt = params.negative_prompt.clone();
    let seed = params.seed;
    let batch_count = params.batch_count;
    let low_vram = params.low_vram;
    let cache_mode = params.cache_mode.clone();
    let upscaler = params.upscaler.clone();
    let upscaler_scale = params.upscaler_scale;

    let (config, model_config) = PresetBuilder::default()
        .preset(preset)
        .prompt(&params.prompt)
        .with_modifier(move |(mut config_b, mut model_b)| {
            // Output path
            config_b.output(output);

            // Optional overrides
            if let Some(s) = steps {
                config_b.steps(s);
            }
            if let Some(w) = width {
                config_b.width(w);
            }
            if let Some(h) = height {
                config_b.height(h);
            }
            if let Some(neg) = negative_prompt {
                config_b.negative_prompt(neg);
            }
            config_b.seed(seed as i32);
            config_b.batch_count(batch_count);

            // Preview config (D-02: file-based, PREVIEW_PROJ)
            config_b.preview_output(preview_output);
            config_b.preview_mode(PreviewType::PREVIEW_PROJ);
            config_b.preview_interval(1);

            // Low VRAM optimizations
            if low_vram {
                model_b.vae_tiling(true).flash_attention(true);
                let (new_config, new_model) =
                    lazily_load_params_from_disk((config_b, model_b))?;
                config_b = new_config;
                model_b = new_model;
            }

            // Cache mode
            if let Some(ref cache) = cache_mode {
                match cache.to_uppercase().as_str() {
                    "UCACHE" => {
                        config_b.ucache_caching(UCacheParamsBuilder::default().build().unwrap());
                    }
                    "EASYCACHE" => {
                        config_b.easy_cache_caching(
                            EasyCacheParamsBuilder::default().build().unwrap(),
                        );
                    }
                    "DBCACHE" => {
                        config_b
                            .db_cache_caching(DbCacheParamsBuilder::default().build().unwrap());
                    }
                    "TAYLORSEER" => {
                        config_b.taylor_seer_caching();
                    }
                    "CACHEDIT" => {
                        config_b.cache_dit_caching(
                            DbCacheParamsBuilder::default().build().unwrap(),
                        );
                    }
                    "SPECTRUM" => {
                        config_b.spectrum_caching(
                            SpectrumCacheParamsBuilder::default().build().unwrap(),
                        );
                    }
                    _ => {} // Unknown cache mode: silently ignore
                };
            }

            // Upscaler
            if let Some(ref up) = upscaler {
                let converted = match up.to_lowercase().as_str() {
                    "latent" => Upscaler::SD_HIRES_UPSCALER_LATENT,
                    "latentnearest" | "latent_nearest" => {
                        Upscaler::SD_HIRES_UPSCALER_LATENT_NEAREST
                    }
                    "latentnearestexact" | "latent_nearest_exact" => {
                        Upscaler::SD_HIRES_UPSCALER_LATENT_NEAREST_EXACT
                    }
                    "latentantialiased" | "latent_antialiased" => {
                        Upscaler::SD_HIRES_UPSCALER_LATENT_ANTIALIASED
                    }
                    "latentbicubic" | "latent_bicubic" => {
                        Upscaler::SD_HIRES_UPSCALER_LATENT_BICUBIC
                    }
                    "latentbicubicantialiased" | "latent_bicubic_antialiased" => {
                        Upscaler::SD_HIRES_UPSCALER_LATENT_BICUBIC_ANTIALIASED
                    }
                    "lanczos" => Upscaler::SD_HIRES_UPSCALER_LANCZOS,
                    "nearest" => Upscaler::SD_HIRES_UPSCALER_NEAREST,
                    _ => Upscaler::SD_HIRES_UPSCALER_LATENT, // fallback
                };
                let hires = HiresParamsBuilder::default()
                    .scale(upscaler_scale)
                    .build()
                    .unwrap();
                model_b.hires_params(converted, hires, None);
            }

            Ok((config_b, model_b))
        })
        .build()
        .map_err(|e| anyhow!("{}", e))?;

    Ok((config, model_config))
}
