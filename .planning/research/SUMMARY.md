# Project Research Summary

**Project:** diffusion-rs GUI
**Domain:** Flutter desktop GUI wrapping a Rust AI image-generation library via FFI
**Researched:** 2026-06-18
**Confidence:** MEDIUM

## Executive Summary

diffusion-rs GUI è una applicazione desktop a due pannelli (Flutter) che espone l'intero set di parametri CLI della libreria Rust diffusion-rs tramite interfaccia grafica. Il backend Rust (v0.1.20, ~35 preset, stable-diffusion.cpp sotto) è già feature-complete; il lavoro è interamente nel costruire il layer GUI. L'approccio raccomandato è un build a due fasi stretto: Phase 1 consegna una mock UI completamente funzionale — tutti i 15 campi input, barra di avanzamento, pannello preview, toggle tema — guidata da `MockGenerationService` senza alcuna dipendenza Rust. Phase 2 collega il backend Rust reale via flutter_rust_bridge 2.x sostituendo una singola riga nel provider. Questo disaccoppiamento è il cardine architetturale: permette allo sviluppo UI di procedere alla velocità di iterazione Flutter mentre le build del toolchain Rust/GPU vengono rimandate.

Le decisioni tecnologiche chiave sono definite: Flutter 3.22+/Dart 3.4+ per la UI, flutter_rust_bridge 2.x (versione esatta pinnata uguale tra pub package e binary codegen) per FFI, Yaru 6.x per il design system, Riverpod 2.x per la gestione stato (4 provider: params, lifecycle generazione, progress, tema), e `path_provider` per la gestione cross-platform della temp directory. Il bridge crate `gui/rust/` deve vivere nel proprio workspace Cargo isolato — NON come membro del workspace root — per evitare trigger delle build CMake/GPU durante il codegen FRB.

I rischi dominanti sono tutti concentrati in Phase 2: le chiamate Rust long-running devono essere `async` al confine FRB oppure la UI si blocca; i progress callback arrivano da thread C++ e devono fluire attraverso `StreamSink<T>` per raggiungere il Dart event loop in sicurezza; e i 40+ `.unwrap()` in diffusion-rs rendono obbligatori i wrapper `catch_unwind` e `panic = "abort"` prima di qualsiasi test utente. Due blocchi immediati esistono indipendentemente dalla fase: i campi struct `Progress` in `src/api.rs` sono privati e necessitano `pub`, e un placeholder `token.txt` deve essere committato nella root del repo per sbloccare le fresh build CI.

## Key Findings

### Stack Raccomandato

| Tecnologia | Versione | Ruolo | Rationale |
|------------|----------|-------|-----------|
| Flutter / Dart | 3.22+ / 3.4+ | UI framework | Minimo per stabilità desktop e pattern Dart richiesti da frb 2.x |
| flutter_rust_bridge | 2.7.0 (pub + cargo binary pinnati identicamente) | FFI bridge | Unica soluzione matura di FFI tipizzata Dart/Rust per desktop |
| Yaru + yaru_icons | ^6.1.0 / ^2.3.0 | Design system | Richiesto dallo spec; light/dark built-in; YaruPasswordField per token |
| Riverpod 2.x + @riverpod codegen | latest | State management | AsyncNotifier gestisce il lifecycle di generazione in modo idiomatico |
| window_manager | ^0.4.0 | Window control | Dimensione minima finestra, intercept close per cleanup temp |
| path_provider | ^2.1.4 | Temp dir | Path cross-platform corretti su macOS/Linux/Windows |
| multi_split_view | latest | Layout due pannelli | Drag handle resizable, min area constraints out of the box |

**Regola critica:** il pacchetto `flutter_rust_bridge` su pub.dev e il binary `flutter_rust_bridge_codegen` su crates.io devono essere la stessa versione patch. Una discrepanza è la fonte più comune di errori crittici di codegen.

### Feature Attese

**Table stakes (must have — tutti già in scope):**
- Dropdown preset + sotto-dropdown pesi contestuale (pesi visibili solo se il preset li supporta)
- Prompt (multiline espandibile) + campo negative prompt
- Steps, width, height, batch, seed con bottone dado/randomize
- Cache dropdown, preview dropdown, upscaler dropdown, upscaler_scale (nascosto se upscaler = none)
- Token HuggingFace come campo password con toggle show/hide (YaruPasswordField)
- Toggle low_vram
- Bottone Generate (disabilita tutti gli input durante la generazione)
- Progress bar + contatore step ("Step N / totale")
- Preview live (si aggiorna dal path file temp su ogni evento progress via Key-based cache busting)
- Immagine finale + bottone Salva esplicito con folder picker

**Should have (basso costo, alto valore UX — includere in v1):**
- Scorciatoia Cmd/Ctrl+Enter per generare
- Bottone dado seed (azzera a -1)
- Auto-enforce cache non-none quando upscaler è selezionato (o banner warning inline)
- Bottone "Apri cartella output" dopo il salvataggio

**Defer v2+:**
- History prompt / recall
- Queue generazione / cancel mid-run (richiede segnale di abort C++ non presente nel backend)
- Gallery output / pannello history
- Metadata embedding nel PNG salvato
- Lista preset raggruppata/ricercabile

### Approccio Architetturale

L'architettura ha una singola seam critica: l'interfaccia astratta Dart `GenerationService`. Phase 1 inserisce `MockGenerationService`; Phase 2 inserisce `RustGenerationService`. Ogni altro componente — tutti i 4 provider Riverpod, entrambi i pannelli, `TempDirManager` — è scritto una volta e non cambia tra le fasi.

**Componenti principali:**

```
┌─────────────────────────────────────────────────────────┐
│                    Flutter App                          │
│  ┌──────────────────┐    ┌──────────────────────────┐  │
│  │   LeftPanel       │    │       RightPanel         │  │
│  │ GenerationForm    │    │  PreviewPane + SaveBtn   │  │
│  └────────┬─────────┘    └───────────┬──────────────┘  │
│           │                          │                  │
│  ┌────────▼──────────────────────────▼──────────────┐  │
│  │              Riverpod Providers                   │  │
│  │  GenerationParamsNotifier (15 form fields)        │  │
│  │  GenerationNotifier (AsyncNotifier lifecycle)     │  │
│  │  ProgressNotifier (step/total/previewPath)        │  │
│  │  ThemeNotifier (system/light/dark)                │  │
│  └─────────────────────┬─────────────────────────────┘  │
│                        │                               │
│  ┌─────────────────────▼─────────────────────────────┐  │
│  │         GenerationService (abstract)               │  │
│  │  generate(params) → Stream<GenerationEvent>        │  │
│  └──────┬──────────────────────────┬─────────────────┘  │
│         │ Phase 1                  │ Phase 2            │
│  MockGenerationService    RustGenerationService         │
│  (Stream + Timer fake)    (FRB bridge calls)           │
└─────────────────────────────────────────────────────────┘
                                │ Phase 2 only
                    ┌───────────▼───────────┐
                    │  gui/rust/ (FRB crate) │
                    │  get_presets()         │
                    │  get_weights_for_preset│
                    │  generate_image_stream │
                    │  ↕ GuiParams DTO       │
                    └───────────┬────────────┘
                                │
                    ┌───────────▼───────────┐
                    │   diffusion-rs lib     │
                    │   src/api.rs           │
                    └───────────────────────┘
```

**Regola workspace:** `gui/rust/` NON deve essere membro del workspace root `Cargo.toml`. Il codegen FRB deve poter girare senza triggerare il build CMake/GPU.

### Pitfall Critici

| # | Pitfall | Fase | Priorità |
|---|---------|------|----------|
| 1 | UI freeze da chiamata Rust sincrona | Phase 2 | CRITICA |
| 2 | Rust panic attraverso confine FFI (40+ `.unwrap()`) | Phase 2 | CRITICA |
| 3 | Progress callback da thread C++ — threading non sicuro | Phase 2 | CRITICA |
| 4 | Campi `Progress` struct privati (`step`, `steps`, `time`) | Phase 2 pre-req | CRITICA |
| 5 | `token.txt` dipendenza compile-time — fallisce su fresh checkout | Immediato | ALTA |
| 6 | FRB codegen out-of-sync con Rust API | Phase 2 | ALTA |
| 7 | `gui/rust/` come membro workspace root — triggera GPU build | Phase 2 pre-req | ALTA |
| 8 | Yaru font non dichiarato in pubspec.yaml — layout overflow | Phase 1 | MEDIA |
| 9 | Image cache Flutter senza Key change — preview stale | Phase 1 | MEDIA |
| 10 | Cleanup temp dir non affidabile su crash | Phase 1 design | MEDIA |

## Implications for Roadmap

### Struttura Fasi Suggerita (Coarse — 2 fasi)

**Phase 1: Flutter UI Foundation (Mock Mode)**
- Scaffolding Flutter in `gui/`, setup Yaru + Riverpod, layout a due pannelli
- Tutti i 15 campi form wired ai provider
- `MockGenerationService` con stream-based fake progress + placeholder image
- TempDirManager (design e impl per cleanup alla chiusura)
- Test cross-platform (macOS + Linux + Windows) per Yaru font e layout
- **Deliverable:** App interattiva completa, nessuna dipendenza Rust

**Phase 2: Rust Bridge Wiring**
- Pre-requisiti Rust: `pub` su campi Progress, placeholder `token.txt`, `catch_unwind` wrapper, `panic = "abort"`
- Scaffold `gui/rust/` come workspace Cargo isolato
- FRB codegen pipeline + CI check diff
- `GuiParams → Config/ModelConfig` conversion (studia `cli/src/main.rs` come riferimento canonico)
- `RustGenerationService` con `StreamSink<ProgressEvent>`
- `get_presets()`, `get_weights_for_preset()` — rimuove la lista preset hardcoded da Phase 1
- Test end-to-end su macOS e Linux con model reale
- **Deliverable:** Generazione immagini reale funzionante

### Ordine Build Consigliato (dentro Phase 1)

scaffold → value classes + provider skeleton → layout shells (due pannelli) → form fields → right panel (preview + save) → TempDirManager → smoke test end-to-end mock → cross-platform QA

### Research Flag per Planning

- **Phase 2:** FRB2 StreamSink exact API e thread-safety guarantee — verificare contro FRB2 changelog corrente prima dell'implementazione (conoscenza di training, MEDIUM confidence). La conversione `GuiParams → Config/ModelConfig` è complessa — studiare `cli/src/main.rs` esaustivamente durante il planning.
- **Phase 1:** Pattern Flutter + Riverpod + Yaru ben documentati — research-phase non necessaria.

## Confidence Assessment

| Area | Livello | Note |
|------|---------|------|
| Stack | MEDIUM | frb 2.x e Yaru 6.x stabili; verificare version pins su pub.dev prima di scaffoldare |
| Feature | ALTO | Table stakes derivate da conoscenza diretta di A1111, ComfyUI, InvokeAI, Fooocus |
| Architettura | MEDIUM | Service interface seam alta confidenza; FRB2 StreamSink threading da verificare |
| Pitfall | MEDIUM-ALTO | Panic-across-FFI e token.txt confermati da analisi codebase diretta (ALTO); threading FRB2 da training |

## Sources

### Primarie (ALTO confidence)
- `src/api.rs` — Progress struct (campi privati confermati), gen_img_with_progress, threading callback
- `src/preset.rs` — varianti Preset enum, PresetDiscriminants
- `cli/src/main.rs` — riferimento canonico wiring parametri per conversione GuiParams → Config/ModelConfig
- `.planning/codebase/CONCERNS.md` — unwrap pervasivi, FFI unsafe, token.txt

### Secondarie (MEDIUM confidence)
- flutter_rust_bridge 2.x documentation — FRB2 isolate model, StreamSink API, codegen workflow (training Aug 2025)
- Yaru Flutter package (pub.dev/packages/yaru) — theme API, widget inventory (training Aug 2025)
- Riverpod 2.x AsyncNotifier patterns (training Aug 2025)
- AUTOMATIC1111, ComfyUI, InvokeAI, Fooocus, DiffusionBee, Draw Things — convenzioni UX

### Da validare prima dell'implementazione
- FRB2 StreamSink thread-safety guarantee — verificare FRB2 changelog corrente
- multi_split_view API corrente — verificare pub.dev
- Preset/weights mapping — estrarre da `src/preset.rs` durante Phase 1 planning

---
*Research completata: 2026-06-18*
*Pronto per roadmap: sì*
