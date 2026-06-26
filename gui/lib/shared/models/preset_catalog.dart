/// Default generation parameters for a preset, extracted from preset_builder.rs.
class PresetDefaults {
  final int? steps;
  final int? width;
  final int? height;
  final String? weight;

  const PresetDefaults({this.steps, this.width, this.height, this.weight});
}

/// Hardcoded preset catalog mirroring src/preset.rs (per D-10, MOCK-04).
///
/// Contains all 42 presets with their weight variant mappings derived from
/// the subenum annotations in the Rust source. Phase 2 will replace this
/// with FFI calls to get_presets() and get_weights_for_preset().
class PresetCatalog {
  PresetCatalog._();

  /// Per-preset default steps/width/height extracted from src/preset_builder.rs.
  /// Null fields mean the preset relies on model/backend defaults.
  static const Map<String, PresetDefaults> _defaultsByPreset = {
    'StableDiffusion1_4': PresetDefaults(steps: 20, width: 512, height: 512),
    'StableDiffusion1_5': PresetDefaults(steps: 20, width: 512, height: 512),
    'StableDiffusion2_1': PresetDefaults(steps: 25, width: 768, height: 768),
    'StableDiffusion3Medium': PresetDefaults(
      steps: 30,
      width: 1024,
      height: 1024,
    ),
    'StableDiffusion3_5Medium': PresetDefaults(
      steps: 40,
      width: 1024,
      height: 1024,
    ),
    'StableDiffusion3_5Large': PresetDefaults(
      steps: 28,
      width: 1024,
      height: 1024,
    ),
    'StableDiffusion3_5LargeTurbo': PresetDefaults(
      steps: 4,
      width: 1024,
      height: 1024,
    ),
    'SDXLBase1_0': PresetDefaults(steps: 20, width: 1024, height: 1024),
    'SDTurbo': PresetDefaults(steps: 4, width: 512, height: 512),
    'SDXLTurbo1_0': PresetDefaults(steps: 4, width: 512, height: 512),
    'Flux1Dev': PresetDefaults(
      steps: 28,
      width: 1024,
      height: 1024,
      weight: 'Q2_K',
    ),
    'Flux1Schnell': PresetDefaults(
      steps: 4,
      width: 1024,
      height: 1024,
      weight: 'Q2_K',
    ),
    'Flux1Mini': PresetDefaults(
      steps: 20,
      width: 1024,
      height: 1024,
      weight: 'Q8_0',
    ),
    'JuggernautXL11': PresetDefaults(steps: 20, width: 1024, height: 1024),
    'Chroma': PresetDefaults(
      steps: 20,
      width: 512,
      height: 512,
      weight: 'Q4_0',
    ),
    'NitroSDRealism': PresetDefaults(
      steps: 1,
      width: 1024,
      height: 1024,
      weight: 'Q8_0',
    ),
    'NitroSDVibrant': PresetDefaults(
      steps: 1,
      width: 1024,
      height: 1024,
      weight: 'Q8_0',
    ),
    'DiffInstructStar': PresetDefaults(
      steps: 1,
      width: 1024,
      height: 1024,
      weight: 'Q8_0',
    ),
    'ChromaRadiance': PresetDefaults(
      steps: 20,
      width: 512,
      height: 512,
      weight: 'Q4_0',
    ),
    'SSD1B': PresetDefaults(
      steps: 20,
      width: 1024,
      height: 1024,
      weight: 'F8_E4M3',
    ),
    'Flux2Dev': PresetDefaults(
      steps: 20,
      width: 512,
      height: 512,
      weight: 'Q2_K',
    ),
    'ZImageTurbo': PresetDefaults(
      steps: 9,
      width: 512,
      height: 1024,
      weight: 'Q4_K',
    ),
    'QwenImage': PresetDefaults(
      steps: 20,
      width: 1024,
      height: 1024,
      weight: 'Q2_K',
    ),
    'OvisImage': PresetDefaults(
      steps: 20,
      width: 512,
      height: 512,
      weight: 'Q4_0',
    ),
    'DreamShaperXL2_1Turbo': PresetDefaults(
      steps: 6,
      width: 1024,
      height: 1024,
    ),
    'TwinFlowZImageTurboExp': PresetDefaults(
      steps: 3,
      width: 512,
      height: 1024,
      weight: 'Q4_0',
    ),
    'SDXS512DreamShaper': PresetDefaults(
      steps: 1,
      width: 512,
      height: 512,
      weight: 'F16',
    ),
    'Flux2Klein4B': PresetDefaults(
      steps: 4,
      width: 1024,
      height: 1024,
      weight: 'Q8_0',
    ),
    'Flux2KleinBase4B': PresetDefaults(
      steps: 20,
      width: 1024,
      height: 1024,
      weight: 'Q8_0',
    ),
    'Flux2Klein9B': PresetDefaults(
      steps: 4,
      width: 1024,
      height: 1024,
      weight: 'Q4_0',
    ),
    'Flux2KleinBase9B': PresetDefaults(
      steps: 20,
      width: 1024,
      height: 1024,
      weight: 'Q4_0',
    ),
    'SegmindVega': PresetDefaults(steps: 25, width: 1024, height: 1024),
    'Anima': PresetDefaults(
      steps: 30,
      width: 1024,
      height: 1024,
      weight: 'Q8_0',
    ),
    'Anima2': PresetDefaults(
      steps: 30,
      width: 1024,
      height: 1024,
      weight: 'Q8_0',
    ),
    'ErnieImage': PresetDefaults(
      steps: 20,
      width: 1024,
      height: 1024,
      weight: 'Q4_0',
    ),
    'ErnieImageTurbo': PresetDefaults(
      steps: 8,
      width: 1024,
      height: 1024,
      weight: 'Q4_0',
    ),
    'HiDreamO1ImageDev': PresetDefaults(steps: 20, width: 1024, height: 1024),
    'HiDreamO1Image': PresetDefaults(steps: 20, width: 1024, height: 1024),
    'LongCatImage': PresetDefaults(
      steps: 20,
      width: 512,
      height: 512,
      weight: 'Q4_0',
    ),
    'Lens': PresetDefaults(steps: 20, width: 512, height: 512),
    'LensTurbo': PresetDefaults(steps: 4, width: 512, height: 512),
    'BooguImage': PresetDefaults(steps: 20, width: 512, height: 512),
    'BooguImageTurbo': PresetDefaults(steps: 4, width: 512, height: 512),
    'Krea2': PresetDefaults(steps: 20, width: 512, height: 512, weight: 'Q3_K'),
    'Krea2Turbo': PresetDefaults(
      steps: 4,
      width: 512,
      height: 512,
      weight: 'Q3_K',
    ),
  };

  /// Returns the default steps/width/height for [presetName].
  static PresetDefaults getDefaults(String presetName) {
    return _defaultsByPreset[presetName] ?? const PresetDefaults();
  }
}
