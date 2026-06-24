# diffusion-rs GUI

## What This Is

Una GUI desktop Flutter per diffusion-rs che espone tutte le funzionalità della CLI in un'interfaccia grafica a due pannelli: sinistra per i parametri di generazione, destra per la preview live e l'immagine finale. La GUI comunica con la libreria Rust via flutter_rust_bridge (FFI), include un meccanismo di live preview step-by-step durante l'inferenza, e usa file temporanei puliti alla chiusura dell'app. Il progetto vive nella cartella `/gui` del monorepo diffusion-rs esistente.

## Core Value

L'utente può configurare e avviare una vera generazione di immagini con lo stesso set di opzioni della CLI, senza aprire un terminale, con preview live aggiornata ad ogni step di diffusione.

## Requirements

### Validated

- ✓ Generazione immagini con preset multipli (SD 1.x/2.x, SDXL, SD3, Flux, ecc.) — existing
- ✓ Interfaccia CLI con tutti i parametri di generazione — existing
- ✓ Supporto multi-platform desktop (macOS, Linux, Windows) — existing
- ✓ Download modelli da HuggingFace Hub con token opzionale — existing
- ✓ Preview immagine durante la generazione — existing
- ✓ Upscaler post-generazione (8 modalità) — existing
- ✓ Modalità di caching accelerate (UCACHE, EASYCACHE, DBCACHE, TAYLORSEER, CACHEDIT, SPECTRUM) — existing
- ✓ Generazione batch (Rust backend) — existing
- ✓ Progetto Flutter in `/gui` come sottocartella del monorepo diffusion-rs — v1.0
- ✓ Layout a due pannelli ridimensionabile (left: parametri + controlli; right: preview + immagine finale) — v1.0
- ✓ Dropdown preset (41 varianti da PresetDiscriminants) via Rust FFI — v1.0
- ✓ Dropdown pesi contestuale al preset via Rust FFI — v1.0
- ✓ Tutti i 14 campi CLI attivi nel form (prompt, negative, steps, width, height, cache, preview, upscaler, upscaler_scale, seed, low_vram, token) — v1.0
- ✓ Campo token HuggingFace come campo password (testo oscurato, toggle visibilità) — v1.0
- ✓ Bottone Generate che disabilita tutti gli input durante la generazione — v1.0
- ✓ Barra di avanzamento con contatore step durante la generazione — v1.0
- ✓ Preview live aggiornata ad ogni step di diffusione — v1.0
- ✓ Immagine finale nel pannello destro con bottone Save — v1.0
- ✓ File temporanei per immagini (preview e output), ripuliti alla chiusura dell'app — v1.0
- ✓ Tema visivo Yaru con supporto chiaro/scuro/sistema — v1.0
- ✓ Scorciatoia Cmd/Ctrl+Enter equivalente al bottone Generate — v1.0
- ✓ catch_unwind per panic Rust: mostra AlertDialog invece di crashare l'app — v1.0
- ✓ FRB codegen CI check (diff check per binding sincronizzati) — v1.0
- ✓ FORM-15 warning: upscaler attivo senza cache — v1.0

### Active

- [ ] Batch count field nel form UI (FORM-07) — generazione di N immagini alla volta
- [ ] History prompt con recall degli ultimi N prompt usati (UX-01)
- [ ] Gallery output — pannello che mostra le immagini generate nella sessione corrente (UX-02)
- [ ] Cancellazione generazione in corso — richiede segnale abort nel backend C++ (UX-03)
- [ ] Metadata embedding (parametri di generazione) nel PNG salvato (UX-04)
- [ ] Lista preset raggruppata per famiglia con ricerca (UX-05)
- [ ] UI per download/gestione modelli da HuggingFace (MDL-01, MDL-02, MDL-03)

### Out of Scope

- Mobile (iOS/Android) — GUI desktop only; non compatibile con flutter_rust_bridge su mobile
- Web version — non compatibile con FFI nativa e file system access
- Generazioni concorrenti multiple — backend single-threaded per design
- Image-to-image / ControlNet / LoRA UI — esposti solo indirettamente tramite parametri CLI standard

## Context

**Shipped v1.0** (2026-06-23): ~4,782 LOC project code (3,662 Dart + 1,120 Rust), 189 files changed.

**Tech stack:** Flutter 3.44.x + Dart, flutter_rust_bridge 2.12.0, Yaru 10.2.0, Riverpod 3.x, Cargokit (CocoaPods-based build integration), multi_split_view 3.6.2, file_picker 11.x, path_provider, uuid.

**Known technical debt:**
- FRB Dart binding stubs hand-written (codegen requires full C++ build chain); need regeneration after build environment is available
- Batch count not wired in GUI (backend supports it, UI form does not)

## Constraints

- **Tech stack**: Flutter + Dart per la GUI, flutter_rust_bridge per FFI, Yaru per il design system
- **Struttura**: sottocartella `/gui` nel monorepo — nessun repo separato
- **File temporanei**: tutti i path di output usati dalla GUI puntano a una temp dir, pulita all'uscita dell'app
- **Sequenza**: Fase 1 (mock completo) prima del wiring Rust — consente di sviluppare e testare la UI indipendentemente dalla build Rust
- **Platform**: desktop only (macOS, Linux, Windows) — stesso target del backend Rust

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| flutter_rust_bridge per FFI | Standard de facto per Dart↔Rust su desktop; genera bindings tipizzati automaticamente | ✓ Good — cargokit integra build automatica; FRB 2.x streaming idiomatico |
| Fase 1 mock prima del wiring | Disaccoppia sviluppo UI dal build Rust (lungo e dipendente da GPU) | ✓ Good — UI iterata rapidamente senza build Rust; seam FRB-09 funzionò con una riga |
| Yaru come design system | Aspetto coerente su Linux/macOS/Windows; theme chiaro/scuro built-in | ✓ Good — design system completo con YaruPasswordField e YaruExpansionPanel |
| Temp dir per output immagini | Evita di sporcare il filesystem dell'utente; path puliti e prevedibili | ✓ Good — lifecycle con session UUID; cleanup crash sessions all'avvio |
| Sottocartella /gui nel monorepo | Un unico git, dipendenza Rust sempre aggiornata, CI unificato | ✓ Good — path dep su diffusion-rs sempre in sync |
| gui/rust/ isolato da root workspace | Evita trigger build CMake/GPU quando non necessario | ✓ Good — empty [workspace] in Cargo.toml; nessun side effect sul root workspace |
| GenerationService abstract seam | Single provider swap per Phase 2 (D-08) | ✓ Good — FRB-09 completato sostituendo una riga in generation_provider.dart |
| Exhaustive match arms nel bridge Rust | Compiler error su nuovi preset non mappati | ✓ Good — compile-time safety garantita quando diffusion-rs aggiunge preset |
| Cargokit package name = pod target name | Cargokit costruisce artifact path da package name; deve coincidere con pod target | ✓ Good — `rust_lib_diffusion_rs_gui` corretto; build fallisce altrimenti |
| previewBytes in-memory in GenerationState | Evita I/O file per ogni preview step | ✓ Good — Uint8List? in GenerationState; Image.memory nel pannello |

---
*Last updated: 2026-06-23 after v1.0 milestone*
