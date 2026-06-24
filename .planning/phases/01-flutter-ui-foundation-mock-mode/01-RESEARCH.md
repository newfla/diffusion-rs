# Phase 1: Flutter UI Foundation (Mock Mode) - Research

**Researched:** 2026-06-18
**Domain:** Flutter desktop GUI, Yaru design system, Riverpod state management, mock service architecture
**Confidence:** MEDIUM

## Summary

Phase 1 is a greenfield Flutter desktop project (`gui/`) in the diffusion-rs monorepo. It delivers a complete two-panel GUI with all 15 CLI fields (minus batch), collapsible form sections, mock generation service emitting Stream-based progress, preview placeholder, and image saving -- all with zero Rust/GPU dependencies.

The standard stack centers on Flutter 3.x with the `yaru` package (v10.2.0) for Ubuntu-style theming and widgets, `flutter_riverpod` (v3.3.2) for state management via AsyncNotifier, `multi_split_view` (v3.6.2) for the resizable two-panel layout, `file_picker` (v11.0.2) for save dialogs, and `path_provider` (v2.1.6) for temp directory management. The Yaru package includes all needed widgets: `YaruExpandable` for collapsible sections, `YaruExpansionPanel` as a coordinated accordion container, `YaruLinearProgressIndicator` and `YaruCircularProgressIndicator` for generation progress, and theme creation via `createYaruLightTheme()`/`createYaruDarkTheme()`.

**Primary recommendation:** Use `YaruExpansionPanel` (not individual `YaruExpandable` widgets) for the four collapsible form sections, as it provides automatic dividers, coordinated collapse, and consistent styling. Use `CallbackShortcuts` (not `Shortcuts`+`Actions`) for Cmd/Ctrl+Enter since the shortcut does not need context-dependent behavior. Use Riverpod 3.x `AsyncNotifier` for the generation lifecycle state machine (idle/generating/complete/error).

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Collapsible sections using Yaru expansion panels. No batch field (removed from scope).
- **D-02:** Four sections: Model (preset dropdown + weights dropdown), Generation (prompt multiline, negative, steps, width, height, seed+dice-button), Post-processing (preview dropdown, upscaler dropdown, upscaler_scale field), Advanced (cache dropdown, token password field, low_vram toggle).
- **D-03:** Default state on app launch: Model + Generation expanded, Post-processing + Advanced collapsed.
- **D-04:** Field order within Generation: prompt, negative, steps, width/height, seed (dice button resets to -1).
- **D-05:** FORM-15 warning (upscaler active but cache = None): shown as inline text under the cache dropdown inside Advanced section. No auto-selection of cache; user must choose manually.
- **D-06:** Presets without Weight variants: weights dropdown is visible but disabled (label "N/A" or greyed-out).
- **D-07:** Feature-based folder structure under gui/lib/ (params/, generation/, output/, shared/).
- **D-08:** The Phase 1 to Phase 2 seam: GenerationService abstract class. MockGenerationService for Phase 1, RustGenerationService for Phase 2 via single provider line change.
- **D-09:** Left/right panel communication via shared Riverpod providers only.
- **D-10:** Dart hardcoded preset list replicates all presets from src/preset.rs.
- **D-11:** Weight dropdown labels use human-readable enum string (Q4_K, Q8_0, F16, F32, etc.).
- **D-12:** Initial right panel state: neutral placeholder with large icon + "Configure parameters and press Generate".
- **D-13:** During generation before first preview frame: centered YaruCircularProgressIndicator spinner.
- **D-14:** After image save: image remains visible + SnackBar briefly showing saved path. Panel does not reset.

### Claude's Discretion
- Exact Yaru widget choices for expansion panels, snackbar duration, and spinner size -- Claude picks what is most idiomatic for Yaru.
- Weight dropdown "disabled" state styling when preset has no Weight variants.

### Deferred Ideas (OUT OF SCOPE)
- Batch field (FORM-07): explicitly removed from Phase 1 scope.
- History/recall of last N prompts (v2 UX-01).
- Gallery output panel (v2 UX-02).
- Download progress for models (v2 MDL-01).
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SETUP-01 | Flutter project in gui/ as monorepo subfolder | Flutter project creation with `--platforms` flag; pubspec.yaml desktop config |
| SETUP-02 | Bridge crate in gui/rust/ as isolated Cargo workspace (not root member) | Phase 1 creates the directory structure placeholder only; no Rust build needed |
| SETUP-03 | Empty token.txt placeholder committed in repo root | Direct file creation; no special tooling |
| SETUP-04 | App compiles and runs on macOS, Linux, Windows | `flutter create --platforms=macos,linux,windows`; Yaru supports all three |
| UI-01 | Two-panel layout (left form, right preview) | multi_split_view with Axis.horizontal |
| UI-02 | Resizable panels via drag handle | multi_split_view built-in drag dividers with Area min constraints |
| UI-03 | Light/dark theme with Yaru | createYaruLightTheme() / createYaruDarkTheme() |
| UI-04 | Theme follows system by default | YaruTheme + MediaQuery.platformBrightnessOf(context) or ThemeMode.system |
| UI-05 | Manual theme override toggle (Light/System/Dark) | SegmentedButton or YaruSegmentedEntry with 3 segments |
| FORM-01 | Preset dropdown | DropdownButton<String> with full preset catalog from preset.rs |
| FORM-02 | Weights dropdown (contextual to preset) | DropdownButton<String> filtered by preset selection; disabled when no weights |
| FORM-03 | Multiline prompt field | TextField with maxLines: null, minLines: 3 |
| FORM-04 | Negative prompt field | TextField single line |
| FORM-05 | Steps numeric field | TextField with number input formatters |
| FORM-06 | Width/height numeric fields | Two TextFields in a Row |
| FORM-07 | Batch field | DEFERRED -- not in Phase 1 scope per D-01 |
| FORM-08 | Seed field with dice button | Custom SeedField widget: TextField + IconButton(YaruIcons.casino or Icons.casino) |
| FORM-09 | Cache mode dropdown | DropdownButton with 7 options (None + 6 cache modes) |
| FORM-10 | Preview dropdown | DropdownButton with 3 options (None, Fast, Accurate) |
| FORM-11 | Upscaler dropdown | DropdownButton with 9 options (None + 8 modes) |
| FORM-12 | Upscaler scale field (conditional visibility) | TextField visible only when upscaler != None |
| FORM-13 | Token password field | TextField with obscureText + IconButton toggle |
| FORM-14 | Low VRAM toggle | YaruSwitch or Switch (Yaru-themed) |
| FORM-15 | Warning when upscaler active but cache None | Inline Text widget with colorScheme.error color |
| GEN-01 | Generate button | ElevatedButton triggering generation |
| GEN-02 | Form fields disable during generation | Riverpod state drives enabled/disabled on all fields |
| GEN-03 | Linear progress bar during generation | YaruLinearProgressIndicator with value: step/steps |
| GEN-04 | Step counter text | Text("Step {N} / {total}") updated from stream |
| GEN-05 | Fields re-enable on completion | AsyncNotifier state transition back to idle/complete |
| GEN-06 | Cmd/Ctrl+Enter shortcut | CallbackShortcuts with SingleActivator for both meta and control |
| OUT-01 | Preview during generation | Image.memory or Image.file updated from stream events |
| OUT-02 | Final image on completion | Image.file from temp directory |
| OUT-03 | Image maintains aspect ratio | BoxFit.contain in Image widget |
| OUT-04 | Save button after completion | OutlinedButton or ElevatedButton |
| OUT-05 | Folder picker + PNG save | file_picker saveFile() with default Pictures directory |
| OUT-06 | Default save folder = Pictures | path_provider or Platform-specific pictures path |
| TMP-01 | Temp dir with session ID | path_provider getTemporaryDirectory() + uuid for session |
| TMP-02 | Temp dir cleanup on normal exit | WidgetsBindingObserver.didChangeAppLifecycleState or Zone cleanup |
| TMP-03 | Stale temp dir cleanup on startup | List and delete old session directories on app init |
| MOCK-01 | MockGenerationService with Stream progress | Dart Stream.periodic or async* generator |
| MOCK-02 | Mock completes in ~5 seconds | ~20 steps, each ~250ms delay |
| MOCK-03 | Placeholder image on completion | Bundled asset PNG or programmatically generated solid color image |
| MOCK-04 | Hardcoded preset/weight catalog | Dart enum/class mirroring src/preset.rs (42 presets, 22 weight sub-enums) |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Form parameter state | Client (Flutter) | -- | Pure client-side state management via Riverpod |
| Generation lifecycle (mock) | Client (Flutter) | -- | MockGenerationService runs entirely in Dart; no backend |
| Temp file management | Client (Flutter) | OS filesystem | path_provider accesses OS temp directories; dart:io for file ops |
| Image display/preview | Client (Flutter) | -- | Image widget renders from file or memory bytes |
| File save dialog | Client (Flutter) | OS native dialog | file_picker invokes OS-native save dialog |
| Theme management | Client (Flutter) | OS (system theme) | Yaru theme follows system brightness; manual override in app state |
| Keyboard shortcuts | Client (Flutter) | -- | CallbackShortcuts intercepts key events at widget level |
| Preset/weight catalog | Client (Flutter) | -- | Hardcoded Dart data; Phase 2 replaces with Rust FFI calls |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| yaru | ^10.2.0 | Design system: themes, icons, widgets (expandable, progress, switch) | Official Ubuntu Flutter theme; provides createYaruLightTheme/createYaruDarkTheme, YaruExpandable, YaruExpansionPanel, YaruLinearProgressIndicator, YaruCircularProgressIndicator, YaruIcons [CITED: pub.dev/packages/yaru] |
| flutter_riverpod | ^3.3.2 | State management with AsyncNotifier for generation lifecycle | De facto Flutter state management; AsyncNotifier handles loading/data/error transitions [CITED: pub.dev/packages/flutter_riverpod] |
| multi_split_view | ^3.6.2 | Resizable two-panel layout with drag handle | 353 likes, 25k downloads, supports all platforms, active maintenance [CITED: pub.dev/packages/multi_split_view] |
| file_picker | ^11.0.2 | Save dialog / folder picker on desktop | Official save file dialog support on macOS/Linux/Windows; saveFile() method [CITED: pub.dev/packages/file_picker] |
| path_provider | ^2.1.6 | Temp directory access and Pictures directory | Flutter team package; getTemporaryDirectory() on all desktop platforms [CITED: pub.dev/packages/path_provider] |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| uuid | ^4.5.3 | Generate unique session ID for temp directories | TMP-01: each app session gets unique temp subdirectory [CITED: pub.dev/packages/uuid] |
| riverpod_annotation | ^3.3.2 | Code generation for Riverpod providers (optional) | If using @riverpod annotation syntax instead of manual provider declarations [ASSUMED] |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| multi_split_view | Custom GestureDetector on VerticalDivider | multi_split_view handles min/max constraints, cursor changes, and hit testing out of the box; custom solution needs 100+ lines |
| file_picker | file_selector (Flutter team) | file_picker has broader API (saveFile with default name); file_selector is lower-level |
| flutter_riverpod | provider, bloc | Riverpod locked per CONTEXT.md decision; provides AsyncNotifier which maps cleanly to generation lifecycle |
| CallbackShortcuts | Shortcuts + Actions + Intent | CallbackShortcuts is simpler when shortcut behavior is context-independent (our case) |

**Installation (pubspec.yaml dependencies):**
```yaml
dependencies:
  flutter:
    sdk: flutter
  yaru: ^10.2.0
  flutter_riverpod: ^3.3.2
  multi_split_view: ^3.6.2
  file_picker: ^11.0.2
  path_provider: ^2.1.6
  uuid: ^4.5.3
```

## Package Legitimacy Audit

> The GSD package-legitimacy tool only supports npm/pypi/crates ecosystems. All packages below are from **pub.dev** (Dart/Flutter registry) and were verified manually via WebFetch against pub.dev pages.

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| yaru | pub.dev | 5+ yrs | high (Ubuntu official) | github.com/ubuntu/yaru.dart | OK (manual) | Approved -- official Ubuntu publisher |
| flutter_riverpod | pub.dev | 4+ yrs | very high | github.com/rrousselGit/riverpod | OK (manual) | Approved -- de facto standard |
| multi_split_view | pub.dev | 3+ yrs | ~25k | github.com/caduandrade/multi_split_view | OK (manual) | Approved -- verified publisher, 353 likes |
| file_picker | pub.dev | 5+ yrs | very high | github.com/miguelpruivo/flutter_file_picker | OK (manual) | Approved -- widely used |
| path_provider | pub.dev | 6+ yrs | very high | github.com/flutter/packages | OK (manual) | Approved -- Flutter team package |
| uuid | pub.dev | 8+ yrs | ~9.2M | github.com/Daegalus/dart-uuid | OK (manual) | Approved -- verified publisher |

**Packages removed due to [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none (all packages verified manually on pub.dev; the GSD tool flagged them SUS only because it cannot query pub.dev -- npm-only limitation)

## Architecture Patterns

### System Architecture Diagram

```
                         gui/ Flutter Desktop App
+--------------------------------------------------------------------+
|                                                                      |
|  main.dart                                                           |
|     |                                                                |
|  app.dart (YaruTheme + CallbackShortcuts + MainLayout)              |
|     |                                                                |
|  +------------------+    +--+    +-----------------------------+     |
|  | ParamsPanel      |    |DH|    | OutputPanel                 |     |
|  | (ScrollView)     |    |  |    |                             |     |
|  |                  |    |  |    |  [idle]    -> placeholder    |     |
|  | YaruExpansionPanel|   |  |    |  [loading] -> spinner       |     |
|  |  Model section   |    |  |    |  [progress]-> bar+image    |     |
|  |  Gen section     |    |  |    |  [complete]-> image+save   |     |
|  |  PostProc section|    |  |    |                             |     |
|  |  Advanced section|    |  |    +-----------------------------+     |
|  |                  |    |  |                                        |
|  | [Generate btn]   |    |  |                                        |
|  +------------------+    +--+                                        |
|         |                                                            |
|         v                                                            |
|  Riverpod Providers                                                  |
|  +---------------------+  +------------------------+                |
|  | paramsProvider       |  | generationProvider     |                |
|  | (Notifier<Params>)   |  | (AsyncNotifier<GenSt>) |               |
|  +---------------------+  +------------------------+                |
|                                   |                                  |
|                        GenerationService (abstract)                  |
|                                   |                                  |
|                        MockGenerationService                         |
|                        (Stream<ProgressEvent>)                       |
|                                                                      |
|  +---------------------+  +------------------------+                |
|  | themeProvider        |  | outputProvider         |                |
|  | (Notifier<ThemeMode>)|  | (Notifier<OutputState>)|               |
|  +---------------------+  +------------------------+                |
|                                                                      |
|  TempDirectoryManager (dart:io)                                      |
|    - creates session temp dir on startup                             |
|    - cleans stale sessions on startup                                |
|    - cleans current session on shutdown                              |
+--------------------------------------------------------------------+
```

DH = MultiSplitView drag handle divider

### Recommended Project Structure
```
gui/
  pubspec.yaml
  analysis_options.yaml
  lib/
    main.dart                              # App entry point, Riverpod scope
    app.dart                               # YaruTheme wrapper, MainLayout, shortcuts
    features/
      params/
        params_panel.dart                  # Left panel with scrollable form
        sections/
          model_section.dart               # Preset + Weights dropdowns
          generation_section.dart          # Prompt, neg, steps, w/h, seed
          postproc_section.dart            # Preview, upscaler, scale
          advanced_section.dart            # Cache, warning, token, low_vram
        providers/
          params_provider.dart             # Notifier<ParamsState> with all 15 fields
      generation/
        providers/
          generation_provider.dart         # AsyncNotifier<GenerationState>
        services/
          generation_service.dart          # Abstract GenerationService interface
          mock_generation_service.dart     # Phase 1: Stream-based mock
      output/
        output_panel.dart                  # Right panel: idle/spinner/progress/complete
        providers/
          output_provider.dart             # Notifier<OutputState> (image path, save status)
    shared/
      theme/
        theme_provider.dart                # Notifier<ThemeMode> (light/system/dark)
      widgets/
        seed_field.dart                    # Numeric input + dice icon
        drag_handle.dart                   # MultiSplitView divider builder
      models/
        preset_catalog.dart                # Hardcoded preset enum + weight mappings
        progress_event.dart                # ProgressEvent class (step, steps, time, imageBytes?)
      services/
        temp_directory_manager.dart        # Session temp dir creation/cleanup
  assets/
    placeholder.png                        # Placeholder image for mock completion
  linux/
  macos/
  windows/
  rust/                                    # Phase 2: flutter_rust_bridge crate (placeholder dir)
```

### Pattern 1: GenerationService Abstraction (Phase 1/2 Seam)
**What:** Abstract class defining the generation contract; MockGenerationService implements it for Phase 1
**When to use:** Always -- this is the architectural seam for Phase 2 swap
**Example:**
```dart
// Source: CONTEXT.md D-08
abstract class GenerationService {
  Stream<ProgressEvent> generate(GenerationParams params);
}

class ProgressEvent {
  final int step;
  final int steps;
  final double time;
  final Uint8List? previewImage; // null until preview is available

  ProgressEvent({required this.step, required this.steps, required this.time, this.previewImage});

  bool get isComplete => step >= steps;
}

class MockGenerationService implements GenerationService {
  @override
  Stream<ProgressEvent> generate(GenerationParams params) async* {
    const totalSteps = 20;
    for (var i = 1; i <= totalSteps; i++) {
      await Future.delayed(const Duration(milliseconds: 250));
      yield ProgressEvent(
        step: i,
        steps: totalSteps,
        time: i * 0.25,
        previewImage: null, // mock has no real preview
      );
    }
  }
}
```

### Pattern 2: AsyncNotifier for Generation Lifecycle
**What:** Riverpod AsyncNotifier managing idle/loading/complete/error states
**When to use:** For the generation provider that coordinates form disabling, progress, and output
**Example:**
```dart
// Source: pub.dev/documentation/riverpod AsyncNotifier pattern [CITED: pub.dev/documentation/riverpod/latest/riverpod/AsyncNotifier-class.html]
enum GenerationStatus { idle, generating, complete, error }

class GenerationState {
  final GenerationStatus status;
  final int currentStep;
  final int totalSteps;
  final String? imagePath;
  final String? errorMessage;

  const GenerationState({
    this.status = GenerationStatus.idle,
    this.currentStep = 0,
    this.totalSteps = 0,
    this.imagePath,
    this.errorMessage,
  });
}

class GenerationNotifier extends Notifier<GenerationState> {
  @override
  GenerationState build() => const GenerationState();

  Future<void> generate(GenerationParams params) async {
    state = const GenerationState(status: GenerationStatus.generating);
    final service = ref.read(generationServiceProvider);
    try {
      await for (final event in service.generate(params)) {
        state = GenerationState(
          status: event.isComplete ? GenerationStatus.complete : GenerationStatus.generating,
          currentStep: event.step,
          totalSteps: event.steps,
          imagePath: event.isComplete ? '/path/to/output.png' : null,
        );
      }
    } catch (e) {
      state = GenerationState(status: GenerationStatus.error, errorMessage: e.toString());
    }
  }
}

final generationProvider = NotifierProvider<GenerationNotifier, GenerationState>(
  GenerationNotifier.new,
);
```

### Pattern 3: YaruExpansionPanel for Collapsible Sections
**What:** Accordion-style collapsible sections with automatic dividers
**When to use:** Left panel form sections (Model, Generation, Post-processing, Advanced)
**Example:**
```dart
// Source: pub.dev/documentation/yaru YaruExpansionPanel [CITED: pub.dev/documentation/yaru/latest/yaru/YaruExpansionPanel-class.html]
YaruExpansionPanel(
  headers: const [
    Text('Model'),
    Text('Generation'),
    Text('Post-processing'),
    Text('Advanced'),
  ],
  children: const [
    ModelSection(),
    GenerationSection(),
    PostprocSection(),
    AdvancedSection(),
  ],
  isInitiallyExpanded: const [true, true, false, false], // D-03
  collapseOnExpand: false, // allow multiple sections open
  placeDividers: true,
)
```

### Pattern 4: CallbackShortcuts for Cmd/Ctrl+Enter
**What:** Simple keyboard shortcut binding without Intent/Actions boilerplate
**When to use:** Global Generate shortcut
**Example:**
```dart
// Source: api.flutter.dev/flutter/widgets/CallbackShortcuts-class.html [CITED: api.flutter.dev/flutter/widgets/CallbackShortcuts-class.html]
CallbackShortcuts(
  bindings: {
    const SingleActivator(LogicalKeyboardKey.enter, meta: true): _onGenerate,    // macOS
    const SingleActivator(LogicalKeyboardKey.enter, control: true): _onGenerate, // Linux/Windows
  },
  child: Focus(
    autofocus: true,
    child: MainLayout(),
  ),
)
```

### Pattern 5: MultiSplitView for Resizable Panels
**What:** Horizontal split with draggable divider, min/max constraints
**When to use:** Main layout splitting left panel (params) and right panel (output)
**Example:**
```dart
// Source: pub.dev/documentation/multi_split_view [CITED: pub.dev/documentation/multi_split_view/latest/multi_split_view/MultiSplitView-class.html]
MultiSplitView(
  axis: Axis.horizontal,
  initialAreas: [
    Area(flex: 2, min: 320),  // Left panel: 40% default, min 320px
    Area(flex: 3, min: 280),  // Right panel: 60% default, min 280px
  ],
  dividerBuilder: (axis, index, resizable, dragging, highlighted, themeData) {
    return DragHandle(isDragging: dragging, isHighlighted: highlighted);
  },
  children: [
    ParamsPanel(),
    OutputPanel(),
  ],
)
```

### Pattern 6: Temp Directory Management
**What:** Session-based temp directory with startup cleanup and shutdown cleanup
**When to use:** TMP-01, TMP-02, TMP-03
**Example:**
```dart
// Source: training knowledge [ASSUMED]
import 'dart:io';
import 'package:path_provider/path_provider.dart';
import 'package:uuid/uuid.dart';

class TempDirectoryManager {
  static const _prefix = 'diffusion_rs_gui_';
  late final Directory _sessionDir;

  String get sessionPath => _sessionDir.path;

  Future<void> initialize() async {
    final tempRoot = await getTemporaryDirectory();
    // TMP-03: Clean stale session directories from previous crashes
    await _cleanStaleSessionDirs(tempRoot);
    // TMP-01: Create new session directory
    final sessionId = const Uuid().v4();
    _sessionDir = Directory('${tempRoot.path}/$_prefix$sessionId');
    await _sessionDir.create(recursive: true);
  }

  Future<void> _cleanStaleSessionDirs(Directory tempRoot) async {
    await for (final entity in tempRoot.list()) {
      if (entity is Directory && entity.path.contains(_prefix)) {
        try {
          await entity.delete(recursive: true);
        } catch (_) {
          // Best effort cleanup
        }
      }
    }
  }

  // TMP-02: Cleanup on normal exit
  Future<void> cleanup() async {
    if (await _sessionDir.exists()) {
      await _sessionDir.delete(recursive: true);
    }
  }
}
```

### Anti-Patterns to Avoid
- **Timer.periodic for mock progress:** Use `async*` Stream generator (per CONTEXT.md). Timer.periodic does not naturally complete and requires manual cleanup.
- **Prop drilling between panels:** Use shared Riverpod providers (per D-09). Never pass callbacks/state through MainLayout.
- **Hardcoding hex colors:** Always use `Theme.of(context).colorScheme.*` or `YaruColors.*` (per UI-SPEC Color Rules).
- **Single YaruExpandable widgets without YaruExpansionPanel:** Use the panel container to get consistent dividers and optional coordinated collapse.
- **Using Navigator for panel states:** The right panel is a single widget with state-driven content, not separate routes.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Resizable split panels | Custom GestureDetector + VerticalDivider with mouse cursor handling | multi_split_view | Handles min/max constraints, cursor changes, divider styling, and hit testing; custom solution is 150+ lines with edge cases |
| File save dialog | dart:io file write with hardcoded path | file_picker saveFile() | OS-native dialog, remembers last location, handles permissions; cross-platform without ifdefs |
| Collapsible sections | Custom AnimatedContainer with toggle state | YaruExpansionPanel | Handles animation, dividers, accordion coordination, Yaru styling consistency |
| Temp directory paths | Hardcoded /tmp paths | path_provider getTemporaryDirectory() | Platform-correct temp path on macOS, Linux, Windows |
| Session UUIDs | Custom random string generation | uuid v4 | Cryptographically random, no collisions, standard format |
| Theme management | Custom dark/light theme data | createYaruLightTheme() / createYaruDarkTheme() | Complete Ubuntu-style themes with correct colors, typography, widget overrides |

**Key insight:** This phase is UI-heavy with no custom algorithms. Every "complex" problem (split panels, file dialogs, theming, progress indicators) has a mature Flutter/Yaru solution. The only custom code is the domain-specific parts: preset catalog, generation state machine, and temp file management.

## Preset Catalog Data (from src/preset.rs)

The Dart mock catalog must replicate these presets. Extracted from the current `src/preset.rs`:

**Presets with Weight variants (22 presets):**

| Preset | Weight Type | Default Weight |
|--------|------------ |---------------|
| Flux1Dev | Flux1Weight | Q2_K |
| Flux1Schnell | Flux1Weight | Q2_K |
| Flux1Mini | Flux1MiniWeight | Q8_0 |
| Chroma | ChromaWeight | Q4_0 |
| NitroSDRealism | NitroSDRealismWeight | Q8_0 |
| NitroSDVibrant | NitroSDVibrantWeight | Q8_0 |
| DiffInstructStar | DiffInstructStarWeight | Q8_0 |
| ChromaRadiance | ChromaRadianceWeight | Q8_0 |
| SSD1B | SSD1BWeight | F8_E4M3 |
| Flux2Dev | Flux2Weight | Q2_K |
| ZImageTurbo | ZImageTurboWeight | Q4_K |
| QwenImage | QwenImageWeight | Q2_K |
| OvisImage | OvisImageWeight | Q4_0 |
| TwinFlowZImageTurboExp | TwinFlowZImageTurboExpWeight | Q4_0 |
| SDXS512DreamShaper | SDXS512DreamShaperWeight | F16 |
| Flux2Klein4B | Flux2Klein4BWeight | Q8_0 |
| Flux2KleinBase4B | Flux2KleinBase4BWeight | Q8_0 |
| Flux2Klein9B | Flux2Klein9BWeight | Q4_0 |
| Flux2KleinBase9B | Flux2KleinBase9BWeight | Q4_0 |
| Anima | AnimaWeight | Q8_0 |
| Anima2 | Anima2Weight | Q8_0 |
| ErnieImage | ErnieImageWeight | Q4_0 |
| ErnieImageTurbo | ErnieImageWeight (shared) | Q4_0 |
| LongCatImage | LongCatImageWeight | Q4_0 |

**Presets without Weight variants (18 presets):**

StableDiffusion1_4, StableDiffusion1_5, StableDiffusion2_1, StableDiffusion3Medium, StableDiffusion3_5Medium, StableDiffusion3_5Large, StableDiffusion3_5LargeTurbo, SDXLBase1_0, SDTurbo, SDXLTurbo1_0, JuggernautXL11, DreamShaperXL2_1Turbo, SegmindVega, HiDreamO1ImageDev, HiDreamO1Image, Lens, LensTurbo

**Total: 42 presets** (24 with weights, 18 without). Note: ErnieImage and ErnieImageTurbo share the same ErnieImageWeight type.

**Weight types per sub-enum** (from subenum annotations in preset.rs):

The weight variants available for each preset are defined by the `#[subenum(...)]` annotations. Each weight type is included in specific sub-enums. The full mapping is derived from the source code and must be replicated exactly in the Dart catalog. [VERIFIED: codebase grep of src/preset.rs]

## Common Pitfalls

### Pitfall 1: YaruTheme Initialization
**What goes wrong:** App shows default Material theme instead of Yaru styling
**Why it happens:** Forgetting to wrap the app with `YaruTheme` or using `MaterialApp` without passing the Yaru theme data
**How to avoid:** Use `YaruTheme(builder: (context, yaru, child) => MaterialApp(theme: yaru.theme, darkTheme: yaru.darkTheme, ...))` as the outermost wrapper
**Warning signs:** Orange/blue Material accent colors instead of Ubuntu orange; wrong font

### Pitfall 2: MultiSplitView Without Focus Scope
**What goes wrong:** Keyboard shortcuts stop working when user drags the split handle
**Why it happens:** MultiSplitView's divider captures focus, and the CallbackShortcuts binding loses its focus scope
**How to avoid:** Wrap the entire layout in a `Focus(autofocus: true, ...)` widget above the CallbackShortcuts; or place shortcuts at the Scaffold level
**Warning signs:** Cmd/Ctrl+Enter works initially but stops after interacting with the divider

### Pitfall 3: Stream Not Cancelling on Widget Dispose
**What goes wrong:** Mock generation stream continues emitting after user navigates away or hot-reloads
**Why it happens:** Riverpod notifier subscribes to stream but does not cancel when provider is disposed
**How to avoid:** Use `ref.onDispose()` to cancel stream subscriptions in the generation notifier; or use `await for` which naturally handles generator cleanup
**Warning signs:** Multiple simultaneous generations running, state corruption

### Pitfall 4: file_picker Platform Dependencies on Linux
**What goes wrong:** file_picker crashes or shows empty dialog on Linux
**Why it happens:** file_picker on Linux requires `zenity` or `kdialog` installed; missing on minimal Linux installs
**How to avoid:** Document the dependency; add a try/catch around saveFile() with a fallback error message
**Warning signs:** PlatformException on Linux only

### Pitfall 5: Temp Directory Permissions on Windows
**What goes wrong:** App fails to create or delete temp directories on Windows
**Why it happens:** Windows temp directory paths are user-specific and may have long path limitations
**How to avoid:** Use path_provider (returns correct per-user temp path); keep filenames short; handle IOException
**Warning signs:** FileSystemException with "access denied" on Windows

### Pitfall 6: DropdownButton State Not Updating
**What goes wrong:** Selecting a new preset does not update the weights dropdown
**Why it happens:** Weights dropdown value is stale because the provider holding weights list was not rebuilt when preset changed
**How to avoid:** Make weights dropdown a computed/derived value from the selected preset in the params provider; ensure the provider rebuilds when preset changes
**Warning signs:** Weights dropdown shows options from previous preset

### Pitfall 7: Forgotten obscureText Toggle State
**What goes wrong:** Token field visibility toggle does not persist; toggling other sections resets it
**Why it happens:** obscureText state stored in local widget state that gets rebuilt when parent YaruExpansionPanel rerenders
**How to avoid:** Store the toggle state in the Riverpod params provider, not in local StatefulWidget state
**Warning signs:** Clicking another section header resets the token visibility

## Code Examples

### Flutter Project Creation Command
```bash
# Source: Flutter documentation [ASSUMED]
cd /path/to/diffusion-rs
flutter create gui --platforms=macos,linux,windows --project-name=diffusion_rs_gui
```

### App Entry Point (main.dart)
```dart
// Source: training knowledge [ASSUMED]
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'app.dart';

void main() {
  runApp(
    const ProviderScope(
      child: DiffusionRsApp(),
    ),
  );
}
```

### App Root with Yaru Theme (app.dart)
```dart
// Source: pub.dev Yaru documentation [CITED: pub.dev/documentation/yaru/latest/yaru/YaruTheme-class.html]
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:yaru/yaru.dart';

class DiffusionRsApp extends ConsumerWidget {
  const DiffusionRsApp({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final themeMode = ref.watch(themeModeProvider);

    return YaruTheme(
      builder: (context, yaru, child) {
        return MaterialApp(
          title: 'diffusion-rs',
          theme: yaru.theme,
          darkTheme: yaru.darkTheme,
          themeMode: themeMode,
          debugShowCheckedModeBanner: false,
          home: CallbackShortcuts(
            bindings: {
              const SingleActivator(LogicalKeyboardKey.enter, meta: true): () => _onGenerate(ref),
              const SingleActivator(LogicalKeyboardKey.enter, control: true): () => _onGenerate(ref),
            },
            child: Focus(
              autofocus: true,
              child: const MainLayout(),
            ),
          ),
        );
      },
    );
  }

  void _onGenerate(WidgetRef ref) {
    // Delegate to generation provider
  }
}
```

### Save File with file_picker
```dart
// Source: pub.dev/packages/file_picker [CITED: pub.dev/packages/file_picker]
import 'package:file_picker/file_picker.dart';

Future<void> saveImage(String sourcePath, String preset, int seed) async {
  final timestamp = DateTime.now().millisecondsSinceEpoch;
  final fileName = '${preset}_${seed}_$timestamp.png';

  final outputPath = await FilePicker.platform.saveFile(
    dialogTitle: 'Save generated image',
    fileName: fileName,
    type: FileType.image,
    allowedExtensions: ['png'],
  );

  if (outputPath != null) {
    final sourceFile = File(sourcePath);
    await sourceFile.copy(outputPath);
    // Show SnackBar with saved path
  }
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| yaru_widgets separate package | Merged into `yaru` single package | yaru 4.0.0 (March 2024) | Import `package:yaru/yaru.dart` only; no separate yaru_widgets dependency [CITED: pub.dev/packages/yaru/changelog] |
| Riverpod StateNotifier | Riverpod Notifier/AsyncNotifier | Riverpod 2.0 (2023) | StateNotifier is legacy; use Notifier<T> for sync state, AsyncNotifier<T> for async [CITED: pub.dev/packages/flutter_riverpod/changelog] |
| Riverpod 2.x AutoDispose variants | Unified in Riverpod 3.x | Riverpod 3.0.0 (Sept 2025) | AutoDisposeNotifier and Notifier fused; simpler API [CITED: pub.dev/packages/flutter_riverpod/changelog] |
| YaruExpansionPanel `isInitiallyExpanded` (added v8.0.0) | Current stable API | yaru 8.0.0 | Replaces manual per-item expansion state management [CITED: pub.dev/packages/yaru/changelog] |

**Deprecated/outdated:**
- `yaru_widgets` package: merged into `yaru` since v4.0.0. Do not add as a separate dependency.
- `StateNotifier` / `StateNotifierProvider`: legacy Riverpod pattern. Use `Notifier` / `NotifierProvider` or `AsyncNotifier` / `AsyncNotifierProvider`.
- `YaruPasswordField`: Does not exist in Yaru. Use standard `TextField` with `obscureText: true` and an `IconButton` suffix for visibility toggle. [CITED: pub.dev/documentation/yaru/latest/yaru/yaru-library.html]

## Project Constraints (from CLAUDE.md)

- **Tech stack**: Flutter + Dart for GUI, flutter_rust_bridge for FFI (Phase 2 only), Yaru for design system
- **Structure**: subfolder `/gui` in monorepo -- no separate repo
- **Temp files**: all output paths point to temp dir, cleaned at app exit
- **Sequence**: Phase 1 (mock complete) before Rust wiring
- **Platform**: desktop only (macOS, Linux, Windows)
- **GSD Workflow**: Use GSD commands for all file changes

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `flutter create --platforms=macos,linux,windows` creates a desktop-only project | Code Examples | Command flags may differ; need to verify with installed Flutter CLI |
| A2 | riverpod_annotation v3.3.2 matches flutter_riverpod v3.3.2 | Standard Stack / Supporting | Version mismatch would cause compilation errors |
| A3 | `file_picker` on Linux requires `zenity` or `kdialog` | Common Pitfalls | May not apply to all Linux distros; Ubuntu ships zenity by default |
| A4 | `YaruExpansionPanel.collapseOnExpand` defaults to true | Architecture Patterns | If default is false, multiple sections open simultaneously which is actually what we want (D-03 wants Model+Generation both open) |
| A5 | The `async*` generator in MockGenerationService automatically cleans up when the stream subscription is cancelled | Architecture Patterns | If not, need explicit cancellation logic |

## Open Questions

1. **Placeholder image for mock completion (MOCK-03)**
   - What we know: Need an image to display when mock generation completes
   - What's unclear: Should we bundle a real PNG asset or generate a solid-color image programmatically?
   - Recommendation: Bundle a small (~50KB) placeholder PNG in `gui/assets/`. Simpler than programmatic generation and ensures consistent behavior across platforms.

2. **Pictures directory path for default save location (OUT-06)**
   - What we know: path_provider provides `getTemporaryDirectory()` but there is no `getPicturesDirectory()` in the standard API
   - What's unclear: How to reliably get the system Pictures folder on all three platforms
   - Recommendation: Use `getDownloadsDirectory()` from path_provider as fallback, or use environment variables (`$HOME/Pictures` on Linux/macOS, `%USERPROFILE%\Pictures` on Windows). Alternatively, file_picker may default to a sensible directory.

3. **Generate button position: inside or outside scroll view**
   - What we know: UI-SPEC says "pinned at the bottom of the left panel (outside the scroll view)"
   - What's unclear: None -- this is clear from UI-SPEC
   - Recommendation: Use a Column with Expanded(SingleChildScrollView(...)) + Padding(ElevatedButton("Generate"))

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Flutter SDK | All (project creation, build, run) | NO | -- | Must install Flutter SDK before execution |
| Dart SDK | All (comes with Flutter) | NO | -- | Installed with Flutter |
| Xcode Command Line Tools | macOS build | Unknown | -- | Required for macOS; likely present on dev machine |
| CMake | Phase 2 only (Rust build) | Not needed Phase 1 | -- | -- |
| Rust toolchain | Phase 2 only | Not needed Phase 1 | -- | -- |

**Missing dependencies with no fallback:**
- **Flutter SDK**: Must be installed before any Phase 1 work. Install via `brew install --cask flutter` (macOS) or official Flutter installer.

**Missing dependencies with fallback:**
- None -- Flutter SDK is the only required dependency for Phase 1.

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | Not applicable -- no user authentication in the app |
| V3 Session Management | No | Not applicable -- desktop app, no sessions |
| V4 Access Control | No | Not applicable -- single-user desktop app |
| V5 Input Validation | Yes | Numeric field validation (steps, width, height, seed, scale); prevent non-numeric input |
| V6 Cryptography | No | HF token stored in memory only, not persisted (Phase 1) |

### Known Threat Patterns for Flutter Desktop

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Temp file leakage (sensitive images left on disk after crash) | Information Disclosure | TMP-03: cleanup stale sessions on startup; best-effort cleanup on exit |
| HF token in memory | Information Disclosure | Token stored as obscured TextField value; not written to disk or logs in Phase 1 |
| Path traversal in save dialog | Tampering | Use file_picker native dialog (OS-managed, sandboxed) |
| Malicious file name in save | Tampering | Sanitize preset name and seed before building filename (remove special chars) |

## Sources

### Primary (HIGH confidence)
- `src/preset.rs` -- Full Preset enum with 42 variants and 22 weight sub-enums (verified by direct codebase read)
- `src/api.rs` lines 82-88 -- Progress struct fields (step: i32, steps: i32, time: f32) (verified by direct codebase read)

### Secondary (MEDIUM confidence)
- [pub.dev/packages/yaru](https://pub.dev/packages/yaru) -- v10.2.0, widget API verified via documentation pages
- [pub.dev/packages/flutter_riverpod](https://pub.dev/packages/flutter_riverpod) -- v3.3.2, AsyncNotifier API verified
- [pub.dev/packages/multi_split_view](https://pub.dev/packages/multi_split_view) -- v3.6.2, Area class API verified
- [pub.dev/packages/file_picker](https://pub.dev/packages/file_picker) -- v11.0.2, saveFile() on desktop verified
- [pub.dev/packages/path_provider](https://pub.dev/packages/path_provider) -- v2.1.6, getTemporaryDirectory() on desktop verified
- [pub.dev/packages/uuid](https://pub.dev/packages/uuid) -- v4.5.3, verified publisher
- [pub.dev/documentation/yaru/latest/yaru/yaru-library.html](https://pub.dev/documentation/yaru/latest/yaru/yaru-library.html) -- Widget export list verified
- [pub.dev/packages/yaru/changelog](https://pub.dev/packages/yaru/changelog) -- yaru_widgets merge in v4.0.0, YaruExpansionPanel API evolution
- [api.flutter.dev/flutter/widgets/CallbackShortcuts-class.html](https://api.flutter.dev/flutter/widgets/CallbackShortcuts-class.html) -- SingleActivator pattern

### Tertiary (LOW confidence)
- Flutter project creation command (`flutter create --platforms=...`) -- based on training knowledge, not verified against installed CLI

## Metadata

**Confidence breakdown:**
- Standard stack: MEDIUM -- all package versions verified via pub.dev WebFetch but not via official Dart/Flutter tooling (CLI not installed)
- Architecture: MEDIUM -- patterns derived from official documentation and pub.dev API docs; code examples are idiomatic but not tested
- Pitfalls: MEDIUM -- based on training knowledge of Flutter desktop development and Yaru usage patterns

**Research date:** 2026-06-18
**Valid until:** 2026-07-18 (30 days -- Flutter/Yaru ecosystem is stable; major versions unlikely to change)
