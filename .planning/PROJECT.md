# diffusion-rs GUI

## What This Is

Una GUI desktop Flutter per diffusion-rs che espone tutte le funzionalità della CLI in un'interfaccia grafica a due pannelli: sinistra per i parametri di generazione, destra per la preview e l'immagine finale. La GUI comunica con la libreria Rust via flutter_rust_bridge (FFI) e usa file temporanei puliti alla chiusura dell'app. Il progetto vive nella cartella `/gui` del monorepo diffusion-rs esistente.

## Core Value

L'utente può configurare e avviare una generazione di immagini con lo stesso set di opzioni della CLI, senza aprire un terminale.

## Requirements

### Validated

- ✓ Generazione immagini con preset multipli (SD 1.x/2.x, SDXL, SD3, Flux, ecc.) — existing
- ✓ Interfaccia CLI con tutti i parametri di generazione — existing
- ✓ Supporto multi-platform desktop (macOS, Linux, Windows) — existing
- ✓ Download modelli da HuggingFace Hub con token opzionale — existing
- ✓ Preview immagine durante la generazione — existing
- ✓ Upscaler post-generazione (8 modalità) — existing
- ✓ Modalità di caching accelerate (UCACHE, EASYCACHE, DBCACHE, TAYLORSEER, CACHEDIT, SPECTRUM) — existing
- ✓ Generazione batch — existing

### Active

- [ ] Progetto Flutter in `/gui` come sottocartella del monorepo diffusion-rs
- [ ] Layout a due pannelli (left: parametri + controlli; right: preview + immagine finale)
- [ ] Pannello sinistro: dropdown preset, dropdown pesi (contestuale al preset), tutti i campi CLI (prompt, negative, steps, width, height, batch, cache, preview, upscaler, upscaler_scale, seed, low_vram, output folder)
- [ ] Campo token HuggingFace come campo password (testo oscurato, toggle visibilità)
- [ ] Bottone Start che disabilita tutti gli input durante la generazione
- [ ] Barra di avanzamento durante la generazione
- [ ] Pannello destro: visualizzazione preview intermedia, poi immagine finale con bottone Salva
- [ ] File temporanei usati per immagini (preview e output), ripuliti alla chiusura dell'app
- [ ] Tema visivo Yaru (yaru Flutter package)
- [ ] Supporto tema chiaro/scuro: default = sistema, override manuale via toggle
- [ ] Fase 1 — mock mode: UI completa e funzionale, nessuna chiamata all'API Rust (progress bar simulata, immagine placeholder)
- [ ] Fase 2 — wiring: integrazione reale con diffusion-rs via flutter_rust_bridge

### Out of Scope

- Mobile (iOS/Android) — GUI desktop only, non pianificato
- Web version — non compatibile con flutter_rust_bridge su web
- Generazioni concorrenti multiple — una generazione alla volta
- UI di gestione modelli (download, cancellazione) — fuori scope v1
- Image-to-image / ControlNet / LoRA UI — esposti solo indirettamente tramite parametri CLI standard

## Context

Il codice Rust esistente è maturo (v0.1.20, ~30 preset supportati). La CLI (`cli/src/main.rs`) espone 15 parametri rilevanti per la GUI:

| Parametro | Tipo | Note |
|-----------|------|------|
| preset | dropdown | ~35 varianti da PresetDiscriminants |
| weights | dropdown | contestuale al preset, non tutti i preset lo supportano |
| prompt | text area | obbligatorio |
| negative | text field | opzionale |
| steps | int field | opzionale, override del default del preset |
| width / height | int fields | opzionali |
| batch | int field | default 1 |
| output | folder picker | default "./" ma → temp dir nella GUI |
| cache | dropdown | 6 modalità + "nessuno" |
| preview | dropdown | Fast / Accurate / nessuno |
| upscaler | dropdown | 8 modalità + "nessuno" (richiede cache attivo) |
| upscaler_scale | float field | default 2.0, visibile solo se upscaler attivo |
| token | password field | HuggingFace token, toggle visibilità |
| low_vram | toggle | bool |
| seed | int field | -1 = random |

Il dropdown pesi è context-sensitive: appare e cambia le opzioni in base al preset selezionato (alcuni preset non hanno pesi selezionabili).

flutter_rust_bridge è lo standard de facto per FFI Dart↔Rust su desktop.

## Constraints

- **Tech stack**: Flutter + Dart per la GUI, flutter_rust_bridge per FFI, Yaru per il design system
- **Struttura**: sottocartella `/gui` nel monorepo — nessun repo separato
- **File temporanei**: tutti i path di output usati dalla GUI puntano a una temp dir, pulita all'uscita dell'app
- **Sequenza**: Fase 1 (mock completo) prima del wiring Rust — consente di sviluppare e testare la UI indipendentemente dalla build Rust
- **Platform**: desktop only (macOS, Linux, Windows) — stesso target del backend Rust

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| flutter_rust_bridge per FFI | Standard de facto per Dart↔Rust su desktop; genera bindings tipizzati automaticamente | — Pending |
| Fase 1 mock prima del wiring | Disaccoppia sviluppo UI dal build Rust (lungo e dipendente da GPU); permette iterazione veloce | — Pending |
| Yaru come design system | Aspetto coerente su Linux/macOS/Windows; theme chiaro/scuro built-in | — Pending |
| Temp dir per output immagini | Evita di sporcare il filesystem dell'utente; path puliti e prevedibili per la GUI | — Pending |
| Sottocartella /gui nel monorepo | Un unico git, dipendenza Rust sempre aggiornata, CI unificato | — Pending |

## Evolution

Questo documento evolve alle transizioni di fase e ai milestone.

**Dopo ogni fase:**
1. Requisiti validati? → Sposta in Validated con riferimento alla fase
2. Nuovi requisiti emersi? → Aggiungi in Active
3. Decisioni da loggare? → Aggiungi in Key Decisions

**Dopo ogni milestone:**
1. Review completa di tutte le sezioni
2. Core Value ancora corretto?
3. Scope di Out of Scope ancora valido?

---
*Last updated: 2026-06-18 after initialization*
