/// Hardcoded preset catalog mirroring src/preset.rs (per D-10, MOCK-04).
///
/// Contains all 42 presets with their weight variant mappings derived from
/// the subenum annotations in the Rust source. Phase 2 will replace this
/// with FFI calls to get_presets() and get_weights_for_preset().
class PresetCatalog {
  PresetCatalog._();

  /// Ordered list of all 42 preset names matching the Preset enum in
  /// src/preset.rs. Display labels use PascalCase enum names per UI-SPEC.
  static const List<String> presetNames = [
    'StableDiffusion1_4',
    'StableDiffusion1_5',
    'StableDiffusion2_1',
    'StableDiffusion3Medium',
    'StableDiffusion3_5Medium',
    'StableDiffusion3_5Large',
    'StableDiffusion3_5LargeTurbo',
    'SDXLBase1_0',
    'SDTurbo',
    'SDXLTurbo1_0',
    'Flux1Dev',
    'Flux1Schnell',
    'Flux1Mini',
    'JuggernautXL11',
    'Chroma',
    'NitroSDRealism',
    'NitroSDVibrant',
    'DiffInstructStar',
    'ChromaRadiance',
    'SSD1B',
    'Flux2Dev',
    'ZImageTurbo',
    'QwenImage',
    'OvisImage',
    'DreamShaperXL2_1Turbo',
    'TwinFlowZImageTurboExp',
    'SDXS512DreamShaper',
    'Flux2Klein4B',
    'Flux2KleinBase4B',
    'Flux2Klein9B',
    'Flux2KleinBase9B',
    'SegmindVega',
    'Anima',
    'Anima2',
    'ErnieImage',
    'ErnieImageTurbo',
    'HiDreamO1ImageDev',
    'HiDreamO1Image',
    'LongCatImage',
    'Lens',
    'LensTurbo',
  ];

  /// Weight variants available for each preset, derived from subenum
  /// annotations in src/preset.rs. Empty list means no weight variants.
  /// Weight labels use human-readable quantization strings per D-11.
  static const Map<String, List<String>> _weightsByPreset = {
    'StableDiffusion1_4': [],
    'StableDiffusion1_5': [],
    'StableDiffusion2_1': [],
    'StableDiffusion3Medium': [],
    'StableDiffusion3_5Medium': [],
    'StableDiffusion3_5Large': [],
    'StableDiffusion3_5LargeTurbo': [],
    'SDXLBase1_0': [],
    'SDTurbo': [],
    'SDXLTurbo1_0': [],
    'Flux1Dev': ['Q2_K', 'Q3_K', 'Q4_0', 'Q4_K', 'Q8_0'],
    'Flux1Schnell': ['Q2_K', 'Q3_K', 'Q4_0', 'Q4_K', 'Q8_0'],
    'Flux1Mini': ['F32', 'Q2_K', 'Q3_K', 'Q5_K', 'Q6_K', 'Q8_0', 'BF16'],
    'JuggernautXL11': [],
    'Chroma': ['Q4_0', 'Q8_0', 'BF16'],
    'NitroSDRealism': ['F16', 'Q2_K', 'Q3_K', 'Q4_0', 'Q5_0', 'Q6_K', 'Q8_0'],
    'NitroSDVibrant': ['F16', 'Q2_K', 'Q3_K', 'Q4_0', 'Q5_0', 'Q6_K', 'Q8_0'],
    'DiffInstructStar': [
      'F16',
      'Q2_K',
      'Q3_K',
      'Q4_0',
      'Q5_0',
      'Q6_K',
      'Q8_0',
    ],
    'ChromaRadiance': ['Q8_0', 'BF16'],
    'SSD1B': ['F16', 'F8_E4M3'],
    'Flux2Dev': [
      'Q2_K',
      'Q3_K',
      'Q4_0',
      'Q4_1',
      'Q4_K',
      'Q5_0',
      'Q5_1',
      'Q5_K',
      'Q6_K',
      'Q8_0',
      'BF16',
    ],
    'ZImageTurbo': [
      'Q2_K',
      'Q3_K',
      'Q4_0',
      'Q4_K',
      'Q5_0',
      'Q6_K',
      'Q8_0',
      'BF16',
    ],
    'QwenImage': [
      'Q2_K',
      'Q3_K',
      'Q4_0',
      'Q4_1',
      'Q4_K',
      'Q5_0',
      'Q5_1',
      'Q5_K',
      'Q6_K',
      'Q8_0',
      'BF16',
      'F8_E4M3',
    ],
    'OvisImage': ['Q4_0', 'Q8_0', 'BF16'],
    'DreamShaperXL2_1Turbo': [],
    'TwinFlowZImageTurboExp': ['Q3_K', 'Q4_0', 'Q5_0', 'Q6_K', 'Q8_0', 'BF16'],
    'SDXS512DreamShaper': ['F16', 'Q8_0'],
    'Flux2Klein4B': ['Q4_0', 'Q8_0', 'BF16'],
    'Flux2KleinBase4B': ['Q4_0', 'Q8_0', 'BF16'],
    'Flux2Klein9B': ['Q4_0', 'Q8_0', 'BF16'],
    'Flux2KleinBase9B': ['Q4_0', 'Q8_0', 'BF16'],
    'SegmindVega': [],
    'Anima': [
      'Q3_K',
      'Q4_0',
      'Q4_1',
      'Q4_K',
      'Q5_0',
      'Q5_1',
      'Q5_K',
      'Q6_K',
      'Q8_0',
      'BF16',
    ],
    'Anima2': ['Q4_K', 'Q5_K', 'Q6_K', 'Q8_0', 'BF16'],
    'ErnieImage': [
      'F16',
      'Q2_K',
      'Q3_K',
      'Q4_0',
      'Q4_1',
      'Q4_K',
      'Q5_0',
      'Q5_1',
      'Q5_K',
      'Q6_K',
      'Q8_0',
      'BF16',
    ],
    'ErnieImageTurbo': [
      'F16',
      'Q2_K',
      'Q3_K',
      'Q4_0',
      'Q4_1',
      'Q4_K',
      'Q5_0',
      'Q5_1',
      'Q5_K',
      'Q6_K',
      'Q8_0',
      'BF16',
    ],
    'HiDreamO1ImageDev': [],
    'HiDreamO1Image': [],
    'LongCatImage': [
      'Q3_K',
      'Q4_0',
      'Q4_1',
      'Q4_K',
      'Q5_0',
      'Q5_1',
      'Q5_K',
      'Q6_K',
      'Q8_0',
      'BF16',
    ],
    'Lens': [],
    'LensTurbo': [],
  };

  /// Default weight for each preset that has weight variants.
  /// Derived from the (default) annotation in subenum definitions.
  static const Map<String, String> _defaultWeights = {
    'Flux1Dev': 'Q2_K',
    'Flux1Schnell': 'Q2_K',
    'Flux1Mini': 'Q8_0',
    'Chroma': 'Q4_0',
    'NitroSDRealism': 'Q8_0',
    'NitroSDVibrant': 'Q8_0',
    'DiffInstructStar': 'Q8_0',
    'ChromaRadiance': 'Q8_0',
    'SSD1B': 'F8_E4M3',
    'Flux2Dev': 'Q2_K',
    'ZImageTurbo': 'Q4_K',
    'QwenImage': 'Q2_K',
    'OvisImage': 'Q4_0',
    'TwinFlowZImageTurboExp': 'Q4_0',
    'SDXS512DreamShaper': 'F16',
    'Flux2Klein4B': 'Q8_0',
    'Flux2KleinBase4B': 'Q8_0',
    'Flux2Klein9B': 'Q4_0',
    'Flux2KleinBase9B': 'Q4_0',
    'Anima': 'Q8_0',
    'Anima2': 'Q8_0',
    'ErnieImage': 'Q4_0',
    'ErnieImageTurbo': 'Q4_0',
    'LongCatImage': 'Q4_0',
  };

  /// Returns weight variants available for the given [presetName].
  /// Returns an empty list when the preset has no weight variants.
  static List<String> getWeights(String presetName) {
    return _weightsByPreset[presetName] ?? const [];
  }

  /// Returns the default weight for the given [presetName], or null
  /// if the preset has no weight variants.
  static String? getDefaultWeight(String presetName) {
    return _defaultWeights[presetName];
  }

  /// Whether the given [presetName] has weight variants.
  static bool hasWeights(String presetName) {
    final weights = _weightsByPreset[presetName];
    return weights != null && weights.isNotEmpty;
  }
}
