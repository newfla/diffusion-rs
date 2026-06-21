---
phase: 02-rust-bridge-wiring
verified: 2026-06-21T18:30:00Z
status: gaps_found
score: 5/9 must-haves verified
behavior_unverified: 1
overrides_applied: 0
gaps:
  - truth: "Preset dropdown populated by get_presets() Rust (not hardcoded Dart list); weight dropdown updates via get_weights_for_preset()"
    status: failed
    reason: "model_section.dart usa PresetCatalog.presetNames hardcoded. Le funzioni Rust get_presets() e get_weights_for_preset() sono implementate in gui/rust/src/api.rs ma non sono chiamate dalla UI — la UI usa ancora PresetCatalog (lista Dart statica)."
    artifacts:
      - path: "gui/lib/features/params/sections/model_section.dart"
        issue: "Riga 40: items: PresetCatalog.presetNames — nessuna chiamata a getPresets() FFI"
      - path: "gui/lib/features/params/providers/params_provider.dart"
        issue: "Righe 111-115: build() usa PresetCatalog.presetNames.first e PresetCatalog.getDefaultWeight()"
    missing:
      - "model_section.dart deve chiamare getPresets() dal binding FRB invece di PresetCatalog.presetNames"
      - "model_section.dart deve chiamare getWeightsForPreset(preset) invece di PresetCatalog.getWeights()"
      - "ParamsNotifier.build() deve inizializzare selectedPreset dal primo elemento di getPresets()"
  - truth: "FRB codegen integrato nel Flutter build (D-09 waiva CI diff check)"
    status: failed
    reason: "flutter_rust_bridge_codegen generate non e' mai stato eseguito con successo. I file Dart in gui/lib/src/rust/ sono stub scritti manualmente (dichiarato nel SUMMARY), non output codegen reale. RustLibWire in frb_generated.io.dart non ha metodi wire concreti (wire_get_presets, wire_generate_image_stream, ecc.) che FRB vero genera. pdeCallFfi chiama funcId numerici senza wire symbols — al runtime la shared library non sara' trovabile/usabile con questo binding. Il SUMMARY stesso dichiara: 'stubs are type-correct and will be replaced by actual codegen output' e aggiunge setup step obbligatori al developer."
    artifacts:
      - path: "gui/lib/src/rust/frb_generated.io.dart"
        issue: "RustLibWire ha solo _lookup ma nessun metodo wire_* concreto — binding incompleto per FFI"
      - path: "gui/lib/src/rust/frb_generated.dart"
        issue: "pdeCallFfi(funcId: 1/2/3/4) chiama index numerici senza wire symbols — dipende da codegen reale per mappare ai simboli nativi"
    missing:
      - "Eseguire flutter_rust_bridge_codegen generate dopo build C++ di diffusion-rs-sys per produrre binding reali"
      - "RustLibWire deve avere metodi wire_* con chiamate ffi.NativeFunction per ogni entry point Rust"
  - truth: "Rust panic durante la generazione non crasha la GUI: caught by catch_unwind, UI re-enables, errore leggibile mostrato"
    status: failed
    reason: "Solo generate_image_stream ha catch_unwind. get_presets() e get_weights_for_preset() sono annotati #[flutter_rust_bridge::frb(sync)] ma NON hanno catch_unwind wrapper. Un panic in get_presets() o get_weights_for_preset() non e' catturato. FRB-06 richiede 'Tutti gli entry point FFI' abbiano catch_unwind. La nota del piano ('All FFI entry points wrapped in catch_unwind') non e' rispettata per le due funzioni sync."
    artifacts:
      - path: "gui/rust/src/api.rs"
        issue: "get_presets() (riga 47) e get_weights_for_preset() (riga 59) non hanno catch_unwind wrapper"
    missing:
      - "Aggiungere catch_unwind wrapper a get_presets() e get_weights_for_preset() in gui/rust/src/api.rs"
  - truth: "flutter analyze passa (Cargokit errors in rust_builder/cargokit)"
    status: failed
    reason: "flutter analyze (senza filtro lib/) trova 68 issues inclusi 15+ errori in rust_builder/cargokit/build_tool/ (undefined methods, missing imports). Sebbene 'flutter analyze lib/' passi senza errori, la suite completa non passa — Cargokit bundled in rust_builder ha dipendenze non soddisfatte (ed25519_edwards, http non installati nel build_tool)."
    artifacts:
      - path: "gui/rust_builder/cargokit/build_tool/lib/src/precompile_binaries.dart"
        issue: "15 errori: CreateReleaseAsset, verify, Release, RepositoriesService non definiti"
      - path: "gui/rust_builder/cargokit/build_tool/lib/src/verify_binaries.dart"
        issue: "Missing package imports: ed25519_edwards, http"
    missing:
      - "Eseguire flutter pub get in rust_builder/cargokit/build_tool/ oppure escludere cargokit da flutter analyze"

behavior_unverified_items:
  - truth: "Premendo Generate mostra live preview per step e immagine finale reale da diffusion-rs"
    test: "Eseguire la GUI con un preset reale, inserire un prompt, premere Generate"
    expected: "Progress events dal backend Rust appaiono nel pannello destro; immagini PNG intermedie aggiornate per step; immagine finale PNG reale a completamento"
    why_human: "Il comportamento runtime richiede il Rust nativo compilato e il codegen FRB eseguito — non verificabile con grep/analisi statica. I binding Dart sono stub manuali; la pipe RustGenerationService->generateImageStream->GuiProgressEvent e' strutturalmente completa ma non puo' essere provata senza la .dylib/.so compilata."
---

# Phase 2: Rust Bridge Wiring — Verifica

**Goal della Fase:** L'utente puo' avviare una vera generazione di immagini con diffusion-rs direttamente dalla GUI, con preview live aggiornata ad ogni step e immagine finale reale — nessun mock.
**Verificato:** 2026-06-21T18:30:00Z
**Status:** gaps_found
**Re-verification:** No — verifica iniziale

---

## Risultato sintetico

La fase ha prodotto un'impalcatura strutturalmente corretta ma **incompleta su 4 fronti bloccanti**. Il lato Rust (gui/rust/) e' ben implementato: GuiParams DTO, get_presets(), get_weights_for_preset(), generate_image_stream() con catch_unwind e relay-thread pattern sono tutti presenti e compilabili. Il lato Dart (generazione, errori, output panel) e' cablato correttamente attraverso RustGenerationService, il provider swap, il dialog di errore. I gap sono:

1. **Il dropdown preset/weights nella UI usa ancora PresetCatalog hardcoded** (Success Criterion 1 MANCATO).
2. **FRB codegen non e' mai stato eseguito**: i binding Dart sono stub manuali privi di wire symbols concreti — l'app non puo' girare senza il passo manuale richiesto al developer.
3. **catch_unwind manca su get_presets() e get_weights_for_preset()** (FRB-06 parzialmente soddisfatto).
4. **flutter analyze completo fallisce** per 68 issues nel Cargokit bundled.

---

## Observable Truths

| # | Truth | Status | Evidenza |
|---|-------|--------|----------|
| 1 | Preset dropdown popolato da get_presets() Rust; weight dropdown da get_weights_for_preset() | FAILED | model_section.dart:40 usa PresetCatalog.presetNames; get_presets() non e' mai chiamato dalla UI |
| 2 | Premendo Generate mostra live preview per step e immagine finale reale da diffusion-rs | PRESENT_BEHAVIOR_UNVERIFIED | RustGenerationService, output panel, previewBytes wiring presenti e completi; non verificabile senza native library compilata |
| 3 | Rust panic non crasha la GUI: catch_unwind su tutti gli FFI entry point, errore modale, form ri-abilitato | FAILED (PARZIALE) | catch_unwind presente SOLO in generate_image_stream; get_presets() e get_weights_for_preset() non coperti |
| 4 | FRB codegen integrato nel build Flutter (CI diff check waivato per D-09) | FAILED | flutter_rust_bridge_codegen generate mai eseguito; frb_generated.io.dart senza wire methods concreti; developer setup step obbligatori documentati nel SUMMARY |

**Score: 5/9 requisiti FRB verificati (FRB-03,04,05,07,09 PASS; FRB-01,02,06,08 FAIL/PARZIALE)**
**Behavior-unverified: 1**

---

## Verifica Success Criteria

### SC-1: Preset dropdown da get_presets() Rust

**STATUS: FAILED (BLOCKER)**

- `gui/rust/src/api.rs:47-52` — `get_presets()` implementata correttamente con `PresetDiscriminants::VARIANTS`.
- `gui/rust/src/api.rs:59-126` — `get_weights_for_preset()` implementata con match exhaustivo su tutti i preset.
- `gui/lib/src/rust/api/api.dart:82-88` — binding Dart `getPresets()` e `getWeightsForPreset()` presenti.
- **PROBLEMA:** `gui/lib/features/params/sections/model_section.dart:40` — il dropdown usa `PresetCatalog.presetNames` (lista statica Dart). `getPresets()` non e' mai chiamata dalla UI. Stessa situazione per i pesi: `model_section.dart:22-23` usa `PresetCatalog.hasWeights()` e `PresetCatalog.getWeights()`.
- `gui/lib/shared/models/preset_catalog.dart:14` — commento nel file: "Phase 2 will replace this with FFI calls to get_presets() and get_weights_for_preset()" — il rimpiazzo non e' avvenuto.

### SC-2: Live preview e immagine finale da diffusion-rs

**STATUS: PRESENT_BEHAVIOR_UNVERIFIED**

Il wiring strutturale e' completo:
- `gui/lib/features/generation/services/rust_generation_service.dart` — converte params Map in GuiParams DTO, chiama `generateImageStream(params: guiParams)`, itera gli eventi.
- `gui/lib/features/generation/providers/generation_provider.dart:113-120` — popola `previewBytes` in `GenerationState` con i bytes del preview.
- `gui/lib/features/output/output_panel.dart:133-139` — `Image.memory(state.previewBytes!)` visualizza il preview.
- `gui/lib/features/output/output_panel.dart:104-121` — "Downloading model..." quando `currentStep == 0`.
- Il comportamento runtime (immagini reali, step progress) non e' verificabile senza la native library compilata da `flutter_rust_bridge_codegen generate`.

### SC-3: Rust panic non crasha la GUI

**STATUS: FAILED (PARZIALE)**

- `gui/rust/src/api.rs:141-211` — `generate_image_stream()` ha `std::panic::catch_unwind(std::panic::AssertUnwindSafe(...))` che copre tutto il lavoro di generazione.
- `gui/rust/src/api.rs:47` — `get_presets()` annotata `#[flutter_rust_bridge::frb(sync)]` ma **nessun catch_unwind**.
- `gui/rust/src/api.rs:59` — `get_weights_for_preset()` stessa situazione.
- Il piano (02-01-PLAN.md) richiede: "All FFI entry points are wrapped in catch_unwind for defense-in-depth". Non rispettato.
- Il re-enable del form su errore e' wired: `generation_provider.dart:123-128` imposta `GenerationStatus.error` nel catch, e `output_panel.dart:38-50` triggera `showErrorDialog` via listenManual.

### SC-4: FRB codegen integrato nel build

**STATUS: FAILED (PARZIALE — D-09 waiva CI diff check, ma Cargokit build integration e' incompleta)**

- `gui/flutter_rust_bridge.yaml` esiste con `rust_input: crate::api`, `dart_output: lib/src/rust`.
- `gui/rust_builder/` con struttura Cargokit completa (CMake, Xcode hooks, platform directories).
- **PROBLEMA CRITICO:** `flutter_rust_bridge_codegen generate` non e' mai stato eseguito. Il SUMMARY lo ammette esplicitamente: "stubs are type-correct and will be replaced by actual codegen output when the developer runs codegen after the first successful C++ build". `RustLibWire` in `frb_generated.io.dart:129-140` non ha metodi wire — un FRB reale genererebbe metodi come `wire_get_presets`, `wire_generate_image_stream` con signature ffi.NativeFunction. Il `pdeCallFfi(funcId: 1)` in frb_generated.dart dipende da questi symbols.
- D-09 waiva il CI diff check — accettato. Ma D-08 richiede che codegen sia "integrato nel Flutter build" e che "developers non abbiano bisogno di un passo manuale". Il SUMMARY contraddice questo: elenca 3 step manuali obbligatori per il developer.

---

## Verifica Requisiti FRB-01 — FRB-09

| Requisito | Descrizione | Status | Evidenza |
|-----------|-------------|--------|----------|
| FRB-01 | `get_presets() -> Vec<String>` esposta via FRB | PARTIAL — ORPHANED | Funzione Rust presente e corretta (api.rs:47). Binding Dart presente (api.dart:82). **Mai chiamata dalla UI** — PresetCatalog usato al posto. |
| FRB-02 | `get_weights_for_preset(preset: String) -> Vec<String>` esposta via FRB | PARTIAL — ORPHANED | Funzione Rust presente e corretta (api.rs:59). Binding Dart presente (api.dart:87). **Mai chiamata dalla UI** — PresetCatalog.getWeights() usato. |
| FRB-03 | `generate_image_stream(params: GuiParams, sink: StreamSink<ProgressEvent>)` esposta via FRB | VERIFIED (strutturalmente) | api.rs:140 implementa con two-thread relay, mpsc channel, StreamSink. Binding frb_generated.dart:146 corretto. RustGenerationService la chiama. |
| FRB-04 | `GuiParams` e' DTO frb-compatibile con 17 campi primitivi | VERIFIED | gui_params.rs ha 17 campi String/Option/i32/i64/f32/bool. sse_encode_gui_params in frb_generated.dart serializza tutti i 17 campi. |
| FRB-05 | Campi `step`, `steps`, `time` di Progress in src/api.rs sono `pub` | VERIFIED | src/api.rs:83-87: `pub step: i32`, `pub steps: i32`, `pub time: f32`. |
| FRB-06 | Tutti gli entry point FFI in gui/rust/ hanno catch_unwind | FAILED | catch_unwind SOLO in generate_image_stream. get_presets() e get_weights_for_preset() non coperti. |
| FRB-07 | Profilo release usa `panic = "abort"` in gui/rust/Cargo.toml | VERIFIED | gui/rust/Cargo.toml:17-18: `[profile.release]` con `panic = "abort"`. |
| FRB-08 | CI verifica file codegen aggiornati | WAIVED (D-09) | Waivato esplicitamente dal developer nella discussione. Non implementato e non richiesto. |
| FRB-09 | RustGenerationService sostituisce MockGenerationService con una singola riga nel provider | VERIFIED | generation_provider.dart:141: `return RustGenerationService(ref)`. Commento: "Phase 2: RustGenerationService replaces MockGenerationService (FRB-09)". |

---

## Required Artifacts

| Artifact | Stato | Dettagli |
|----------|-------|----------|
| `gui/rust/Cargo.toml` | VERIFIED | Workspace isolato, path dep su diffusion-rs, panic=abort, flutter_rust_bridge 2.12.0 |
| `gui/rust/src/lib.rs` | VERIFIED | Module declarations, init_app() con frb(init) |
| `gui/rust/src/api.rs` | VERIFIED (con gap FRB-06) | 3 funzioni FRB presenti; catch_unwind manca su 2 di 3 |
| `gui/rust/src/gui_params.rs` | VERIFIED | 17 campi primitivi |
| `gui/rust/src/bridge.rs` | VERIFIED | map_preset() + build_configs() completi |
| `src/api.rs` | VERIFIED | Progress fields pub |
| `gui/lib/features/generation/services/rust_generation_service.dart` | VERIFIED | Implementa GenerationService, converte GuiParams, streamma ProgressEvent |
| `gui/lib/shared/widgets/error_dialog.dart` | VERIFIED | showErrorDialog con "Generation Failed", barrierDismissible: false |
| `gui/lib/features/generation/providers/generation_provider.dart` | VERIFIED | Provider usa RustGenerationService; previewBytes in state |
| `gui/lib/features/output/output_panel.dart` | VERIFIED | "Downloading model..." a step==0; Image.memory per preview; errore dialog trigger |
| `gui/pubspec.yaml` | VERIFIED | flutter_rust_bridge: 2.12.0 presente |
| `gui/lib/src/rust/frb_generated.dart` | STUB (non reale) | Stub manuale — wire symbols non generati; funziona come type scaffold non come binding operativo |

---

## Key Link Verification

| Da | A | Via | Status |
|----|---|-----|--------|
| `rust_generation_service.dart` | `gui/rust/src/api.rs` | `generateImageStream(params: guiParams)` in api.dart | STUB — il binding Dart chiama RustLib.instance.api.crateApiGenerateImageStream; RustLib.instance dipende da native library che richiede codegen |
| `generation_provider.dart` | `rust_generation_service.dart` | `generationServiceProvider` ritorna `RustGenerationService(ref)` | VERIFIED — riga 141 |
| `output_panel.dart` | `error_dialog.dart` | `showErrorDialog(context, next.errorMessage!)` in listenManual | VERIFIED — output_panel.dart:46 |
| `model_section.dart` | `api.dart` (getPresets) | NON COLLEGATO | BROKEN — model_section usa PresetCatalog, mai getPresets() |

---

## Behavioral Spot-Checks

Step 7b: SKIPPED — l'app non ha entry point eseguibili senza native library compilata. `cargo check` per gui/rust/ non e' eseguibile senza il full C++ build chain (stable-diffusion.cpp submodule + CMake). `flutter analyze lib/` passa senza errori. I check comportamentali richiedono runtime con FFI funzionante.

---

## Anti-Pattern Scan

| File | Pattern | Severita' | Impatto |
|------|---------|----------|---------|
| `gui/lib/shared/models/preset_catalog.dart:14` | Commento "Phase 2 will replace this" — non sostituito | WARNING | PresetCatalog rimane la sorgente della UI; il rimpiazzo FFI e' dichiarato ma non implementato |
| `gui/lib/src/rust/frb_generated.dart:1-4` | Header "@generated by flutter_rust_bridge@ 2.12.0" ma e' uno stub manuale | WARNING | Il file e' scritto a mano, non generato; dichiarazione fuorviante |
| `gui/rust/src/frb_generated.rs` | Stub placeholder — StreamSink e' un no-op | INFO | Necessario per cargo check; sara' rimpiazzato da codegen reale |
| `gui/rust_builder/cargokit/` | Cargokit con dipendenze non soddisfatte (flutter analyze: 68 issues) | WARNING | Il build tool non compila; potrebbe non funzionare al primo `flutter build` |

Nessun marker `TBD`, `FIXME`, `XXX` trovato nei file modificati dalla fase.

---

## Human Verification Required

### 1. Generazione immagine reale end-to-end

**Test:** Eseguire `flutter_rust_bridge_codegen generate` nella directory `gui/`, poi `flutter run` su macOS/Linux, selezionare un preset, inserire un prompt breve, premere Generate.
**Expected:** La progress bar avanza per ogni step di diffusion; il pannello destro mostra immagini preview intermedie PNG; al completamento appare l'immagine finale reale (non il placeholder Phase 1).
**Why human:** Richiede native library compilata (.dylib/.so) da codegen reale — non verificabile staticamente.

### 2. Rust panic handling end-to-end

**Test:** Provocare un panic nel backend (es. out-of-memory o preset non disponibile) e osservare il comportamento GUI.
**Expected:** Modal AlertDialog "Generation Failed" appare con il messaggio di errore; il form si ri-abilita dopo OK; l'app non crasha.
**Why human:** Comportamento di stato machine sotto errore runtime — non verificabile senza native runtime.

---

## Gaps Summary

**4 gap bloccanti identificati:**

1. **UI non usa le funzioni FRB Rust per preset/weights** — il Success Criterion principale (SC-1) non e' raggiunto. `getPresets()` e `getWeightsForPreset()` sono implementati e correttamente esportati, ma `model_section.dart` usa ancora `PresetCatalog.presetNames` e `PresetCatalog.getWeights()`. Richiede 3-5 righe di modifica in `model_section.dart` e `params_provider.dart`.

2. **FRB codegen mai eseguito — binding Dart sono stub manuali** — `frb_generated.io.dart:RustLibWire` non ha wire methods; il binding non puo' effettuare FFI call. Richiede che il developer esegua `flutter_rust_bridge_codegen generate` dopo una build C++ completa. Questa e' una precondizione obbligatoria per qualsiasi test runtime.

3. **catch_unwind mancante su get_presets() e get_weights_for_preset()** — FRB-06 e' parzialmente implementato. Se queste funzioni sync panickano (es. per un preset mal formato o un problema di inizializzazione), la GUI crasha senza recovery. Fix semplice: avvolgere il corpo delle due funzioni in `std::panic::catch_unwind`.

4. **flutter analyze (root) fallisce con 68 issues in Cargokit** — non blocca `flutter analyze lib/` (0 issues) ma indica che il build_tool Cargokit bundled ha dipendenze rotte. Potrebbe impattare il primo `flutter build`.

**Root cause comune per gap 1 e 2:** Il SUMMARY documenta che il codegen reale non e' stato eseguito e che i binding Dart sono "stub type-correct". La conseguenza e' che nemmeno il wiring UI->FFI->Rust e' stato verificato in pratica, e il Step successivo obbligatorio (chiamare getPresets() dalla UI) non e' avvenuto.

---

_Verificato: 2026-06-21_
_Verifier: Claude (gsd-verifier)_
