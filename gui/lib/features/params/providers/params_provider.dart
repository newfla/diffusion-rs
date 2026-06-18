import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../../shared/models/preset_catalog.dart';

/// Immutable state class holding all form field values.
///
/// Covers all 14 active fields (batch excluded per D-01).
/// The tokenVisible flag is stored here rather than in local widget
/// state so it survives section rebuilds (per RESEARCH.md Pitfall 7).
class ParamsState {
  final String selectedPreset;
  final String? selectedWeight;
  final String prompt;
  final String negativePrompt;
  final int? steps;
  final int? width;
  final int? height;
  final int seed;
  final String cacheMode;
  final String previewMode;
  final String upscalerMode;
  final double upscalerScale;
  final String token;
  final bool lowVram;
  final bool tokenVisible;

  const ParamsState({
    required this.selectedPreset,
    this.selectedWeight,
    this.prompt = '',
    this.negativePrompt = '',
    this.steps,
    this.width,
    this.height,
    this.seed = -1,
    this.cacheMode = 'None',
    this.previewMode = 'None',
    this.upscalerMode = 'None',
    this.upscalerScale = 2.0,
    this.token = '',
    this.lowVram = false,
    this.tokenVisible = false,
  });

  ParamsState copyWith({
    String? selectedPreset,
    String? Function()? selectedWeightFn,
    String? prompt,
    String? negativePrompt,
    int? Function()? stepsFn,
    int? Function()? widthFn,
    int? Function()? heightFn,
    int? seed,
    String? cacheMode,
    String? previewMode,
    String? upscalerMode,
    double? upscalerScale,
    String? token,
    bool? lowVram,
    bool? tokenVisible,
  }) {
    return ParamsState(
      selectedPreset: selectedPreset ?? this.selectedPreset,
      selectedWeight:
          selectedWeightFn != null ? selectedWeightFn() : selectedWeight,
      prompt: prompt ?? this.prompt,
      negativePrompt: negativePrompt ?? this.negativePrompt,
      steps: stepsFn != null ? stepsFn() : steps,
      width: widthFn != null ? widthFn() : width,
      height: heightFn != null ? heightFn() : height,
      seed: seed ?? this.seed,
      cacheMode: cacheMode ?? this.cacheMode,
      previewMode: previewMode ?? this.previewMode,
      upscalerMode: upscalerMode ?? this.upscalerMode,
      upscalerScale: upscalerScale ?? this.upscalerScale,
      token: token ?? this.token,
      lowVram: lowVram ?? this.lowVram,
      tokenVisible: tokenVisible ?? this.tokenVisible,
    );
  }

  /// Converts the params state to a map for passing to the generation service.
  Map<String, dynamic> toMap() {
    return {
      'preset': selectedPreset,
      'weight': selectedWeight,
      'prompt': prompt,
      'negativePrompt': negativePrompt,
      'steps': steps,
      'width': width,
      'height': height,
      'seed': seed,
      'cacheMode': cacheMode,
      'previewMode': previewMode,
      'upscalerMode': upscalerMode,
      'upscalerScale': upscalerScale,
      'token': token,
      'lowVram': lowVram,
    };
  }
}

/// Riverpod Notifier managing all form field state.
///
/// Provides setter methods for each field. The setPreset method
/// automatically resets selectedWeight to the default weight for
/// the new preset (or null if no weights available).
class ParamsNotifier extends Notifier<ParamsState> {
  @override
  ParamsState build() {
    final firstPreset = PresetCatalog.presetNames.first;
    return ParamsState(
      selectedPreset: firstPreset,
      selectedWeight: PresetCatalog.getDefaultWeight(firstPreset),
    );
  }

  void setPreset(String preset) {
    state = state.copyWith(
      selectedPreset: preset,
      selectedWeightFn: () => PresetCatalog.getDefaultWeight(preset),
    );
  }

  void setWeight(String? weight) {
    state = state.copyWith(selectedWeightFn: () => weight);
  }

  void setPrompt(String prompt) {
    state = state.copyWith(prompt: prompt);
  }

  void setNegativePrompt(String negativePrompt) {
    state = state.copyWith(negativePrompt: negativePrompt);
  }

  void setSteps(int? steps) {
    state = state.copyWith(stepsFn: () => steps);
  }

  void setWidth(int? width) {
    state = state.copyWith(widthFn: () => width);
  }

  void setHeight(int? height) {
    state = state.copyWith(heightFn: () => height);
  }

  void setSeed(int seed) {
    state = state.copyWith(seed: seed);
  }

  void setCacheMode(String mode) {
    state = state.copyWith(cacheMode: mode);
  }

  void setPreviewMode(String mode) {
    state = state.copyWith(previewMode: mode);
  }

  void setUpscalerMode(String mode) {
    state = state.copyWith(upscalerMode: mode);
  }

  void setUpscalerScale(double scale) {
    state = state.copyWith(upscalerScale: scale);
  }

  void setToken(String token) {
    state = state.copyWith(token: token);
  }

  void setLowVram(bool lowVram) {
    state = state.copyWith(lowVram: lowVram);
  }

  void setTokenVisible(bool visible) {
    state = state.copyWith(tokenVisible: visible);
  }
}

/// Provider for the form parameters state.
final paramsProvider = NotifierProvider<ParamsNotifier, ParamsState>(
  ParamsNotifier.new,
);
