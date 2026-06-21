//! FRB-annotated public API functions for the Flutter GUI.
//!
//! This module requires FRB codegen to compile fully — `frb_generated::StreamSink`
//! is a placeholder stub until `flutter_rust_bridge_codegen generate` runs.

use std::fs;
use std::path::PathBuf;
use std::sync::mpsc;

use anyhow::Result;
use strum::VariantNames;

use diffusion_rs::api::{Progress, gen_img_with_progress};
use diffusion_rs::preset::{
    Anima2Weight, AnimaWeight, ChromaRadianceWeight, ChromaWeight, DiffInstructStarWeight,
    ErnieImageWeight, Flux1MiniWeight, Flux1Weight, Flux2Klein4BWeight, Flux2Klein9BWeight,
    Flux2KleinBase4BWeight, Flux2KleinBase9BWeight, Flux2Weight, LongCatImageWeight,
    NitroSDRealismWeight, NitroSDVibrantWeight, OvisImageWeight, PresetDiscriminants,
    QwenImageWeight, SDXS512DreamShaperWeight, SSD1BWeight, TwinFlowZImageTurboExpWeight,
    ZImageTurboWeight,
};

use crate::bridge;
use crate::frb_generated::StreamSink;
use crate::gui_params::GuiParams;

/// Progress event sent from Rust to Dart via StreamSink.
#[derive(Debug, Clone)]
pub struct GuiProgressEvent {
    /// Current inference step
    pub step: i32,
    /// Total inference steps
    pub steps: i32,
    /// Time elapsed for this step
    pub time: f32,
    /// Preview image PNG bytes (None if file not yet available)
    pub preview_image: Option<Vec<u8>>,
    /// Final image PNG bytes (populated only on the completion event)
    pub final_image: Option<Vec<u8>>,
}

/// Return the list of all available preset names.
///
/// Uses `PresetDiscriminants::VARIANTS` from strum to stay in sync with the
/// Rust `Preset` enum automatically (FRB-01). Wrapped in catch_unwind per FRB-06.
#[flutter_rust_bridge::frb(sync)]
pub fn get_presets() -> Vec<String> {
    std::panic::catch_unwind(|| {
        PresetDiscriminants::VARIANTS
            .iter()
            .map(|s| s.to_string())
            .collect()
    })
    .unwrap_or_default()
}

/// Return the valid weight variant names for a given preset.
///
/// For presets without weight options, returns an empty vec.
/// Preset string is case-insensitive (FRB-02). Wrapped in catch_unwind per FRB-06.
#[flutter_rust_bridge::frb(sync)]
pub fn get_weights_for_preset(preset: String) -> Vec<String> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| _get_weights_for_preset(preset)))
        .unwrap_or_default()
}

fn _get_weights_for_preset(preset: String) -> Vec<String> {
    use std::str::FromStr;

    let disc = match PresetDiscriminants::from_str(&preset) {
        Ok(d) => d,
        Err(_) => return Vec::new(),
    };

    macro_rules! weight_variants {
        ($weight_type:ty) => {
            <$weight_type>::VARIANTS
                .iter()
                .map(|s| s.to_string())
                .collect()
        };
    }

    match disc {
        // Presets without weights
        PresetDiscriminants::StableDiffusion1_4
        | PresetDiscriminants::StableDiffusion1_5
        | PresetDiscriminants::StableDiffusion2_1
        | PresetDiscriminants::StableDiffusion3Medium
        | PresetDiscriminants::StableDiffusion3_5Medium
        | PresetDiscriminants::StableDiffusion3_5Large
        | PresetDiscriminants::StableDiffusion3_5LargeTurbo
        | PresetDiscriminants::SDXLBase1_0
        | PresetDiscriminants::SDTurbo
        | PresetDiscriminants::SDXLTurbo1_0
        | PresetDiscriminants::JuggernautXL11
        | PresetDiscriminants::DreamShaperXL2_1Turbo
        | PresetDiscriminants::SegmindVega
        | PresetDiscriminants::HiDreamO1ImageDev
        | PresetDiscriminants::HiDreamO1Image
        | PresetDiscriminants::Lens
        | PresetDiscriminants::LensTurbo => Vec::new(),

        // Presets with weights
        PresetDiscriminants::Flux1Dev | PresetDiscriminants::Flux1Schnell => {
            weight_variants!(Flux1Weight)
        }
        PresetDiscriminants::Flux1Mini => weight_variants!(Flux1MiniWeight),
        PresetDiscriminants::Chroma => weight_variants!(ChromaWeight),
        PresetDiscriminants::NitroSDRealism => weight_variants!(NitroSDRealismWeight),
        PresetDiscriminants::NitroSDVibrant => weight_variants!(NitroSDVibrantWeight),
        PresetDiscriminants::DiffInstructStar => weight_variants!(DiffInstructStarWeight),
        PresetDiscriminants::ChromaRadiance => weight_variants!(ChromaRadianceWeight),
        PresetDiscriminants::SSD1B => weight_variants!(SSD1BWeight),
        PresetDiscriminants::Flux2Dev => weight_variants!(Flux2Weight),
        PresetDiscriminants::ZImageTurbo => weight_variants!(ZImageTurboWeight),
        PresetDiscriminants::QwenImage => weight_variants!(QwenImageWeight),
        PresetDiscriminants::OvisImage => weight_variants!(OvisImageWeight),
        PresetDiscriminants::TwinFlowZImageTurboExp => {
            weight_variants!(TwinFlowZImageTurboExpWeight)
        }
        PresetDiscriminants::SDXS512DreamShaper => weight_variants!(SDXS512DreamShaperWeight),
        PresetDiscriminants::Flux2Klein4B => weight_variants!(Flux2Klein4BWeight),
        PresetDiscriminants::Flux2KleinBase4B => weight_variants!(Flux2KleinBase4BWeight),
        PresetDiscriminants::Flux2Klein9B => weight_variants!(Flux2Klein9BWeight),
        PresetDiscriminants::Flux2KleinBase9B => weight_variants!(Flux2KleinBase9BWeight),
        PresetDiscriminants::Anima => weight_variants!(AnimaWeight),
        PresetDiscriminants::Anima2 => weight_variants!(Anima2Weight),
        PresetDiscriminants::ErnieImage | PresetDiscriminants::ErnieImageTurbo => {
            weight_variants!(ErnieImageWeight)
        }
        PresetDiscriminants::LongCatImage => weight_variants!(LongCatImageWeight),
    }
}

/// Stream image generation progress events to Dart.
///
/// Spawns a background thread that:
/// 1. Maps `GuiParams` to diffusion-rs `Config` + `ModelConfig` via `bridge::build_configs`
/// 2. Creates an `mpsc` channel and spawns a relay thread
/// 3. The relay thread reads each `Progress` event, reads preview PNG bytes from
///    disk (D-03: race accepted — `fs::read` failure yields `None`), and emits
///    a `GuiProgressEvent` through the `StreamSink`
/// 4. The main worker thread calls `gen_img_with_progress` (blocking)
/// 5. After generation, reads the final image bytes and emits a completion event
///
/// All work is wrapped in `catch_unwind` for defense-in-depth (D-07/FRB-06).
pub fn generate_image_stream(params: GuiParams, sink: StreamSink<GuiProgressEvent>) -> Result<()> {
    std::thread::spawn(move || {
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<()> {
                let preview_path = PathBuf::from(&params.preview_output);
                let output_path = PathBuf::from(&params.output);

                // Build Config and ModelConfig from GUI params
                let (config, mut model_config) = bridge::build_configs(&params)?;

                // Create mpsc channel for progress relay
                let (tx, rx) = mpsc::channel::<Progress>();

                // Clone sink for the relay thread
                let relay_sink = sink.clone();
                let relay_preview = preview_path.clone();

                // Relay thread: receives Progress events, reads preview file, emits to StreamSink
                let relay_handle = std::thread::spawn(move || {
                    while let Ok(progress) = rx.recv() {
                        // Read preview image bytes (D-03: race accepted, ok() swallows errors)
                        let preview_bytes = fs::read(&relay_preview).ok();

                        let _ = relay_sink.add(GuiProgressEvent {
                            step: progress.step,
                            steps: progress.steps,
                            time: progress.time,
                            preview_image: preview_bytes,
                            final_image: None,
                        });
                    }
                    // Channel closed — generation complete or errored
                });

                // Blocking generation call. Progress events are sent via tx.
                // tx is dropped when gen_img_with_progress returns, closing the channel.
                let gen_result = gen_img_with_progress(&config, &mut model_config, tx);

                // Wait for relay thread to finish processing all buffered events
                relay_handle.join().ok();

                // Check generation result
                gen_result.map_err(|e| anyhow::anyhow!("{}", e))?;

                // Read the final generated image and emit completion event
                let final_bytes = fs::read(&output_path).ok();
                let _ = sink.add(GuiProgressEvent {
                    step: 0,
                    steps: 0,
                    time: 0.0,
                    preview_image: None,
                    final_image: final_bytes,
                });

                Ok(())
            }));

        match result {
            Ok(Ok(())) => { /* stream completes naturally */ }
            Ok(Err(e)) => {
                let _ = sink.add_error(e);
            }
            Err(panic_info) => {
                let msg = panic_info
                    .downcast_ref::<String>()
                    .map(|s| s.as_str())
                    .or_else(|| panic_info.downcast_ref::<&str>().copied())
                    .unwrap_or("Unknown panic in generation");
                let _ = sink.add_error(anyhow::anyhow!("Panic: {}", msg));
            }
        }
    });

    Ok(())
}
