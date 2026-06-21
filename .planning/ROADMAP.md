# Roadmap: diffusion-rs GUI

**Project:** diffusion-rs GUI
**Core Value:** L'utente può configurare e avviare una generazione di immagini con lo stesso set di opzioni della CLI, senza aprire un terminale.
**Total Phases:** 2
**Requirements:** 46 v1 requirements

---

## Overview

Il progetto si articola in due fasi verticali. Phase 1 consegna una GUI Flutter completa e interattiva in modalità mock — nessuna dipendenza dal toolchain Rust/GPU — consentendo iterazione veloce sulla UX. Phase 2 cablata il bridge flutter_rust_bridge, sostituendo il mock con chiamate reali al backend diffusion-rs tramite un'unica seam architetturale.

## Phases

- [x] **Phase 1: Flutter UI Foundation (Mock Mode)** - GUI completa e interattiva con mock service — zero dipendenze Rust (completed 2026-06-18)
- [x] **Phase 2: Rust Bridge Wiring** - Integrazione reale con diffusion-rs via flutter_rust_bridge FFI (completed 2026-06-21)

## Phase Details

### Phase 1: Flutter UI Foundation (Mock Mode)

**Goal**: L'utente può interagire con una GUI desktop a due pannelli completa — tutti i 15 campi CLI, barra di avanzamento, preview placeholder e salvataggio immagine — senza che nessuna dipendenza Rust/GPU sia presente sulla macchina.
**Mode**: mvp
**Depends on**: Nothing (first phase)
**Requirements**: SETUP-01, SETUP-02, SETUP-03, SETUP-04, UI-01, UI-02, UI-03, UI-04, UI-05, FORM-01, FORM-02, FORM-03, FORM-04, FORM-05, FORM-06, FORM-07, FORM-08, FORM-09, FORM-10, FORM-11, FORM-12, FORM-13, FORM-14, FORM-15, GEN-01, GEN-02, GEN-03, GEN-04, GEN-05, GEN-06, OUT-01, OUT-02, OUT-03, OUT-04, OUT-05, OUT-06, TMP-01, TMP-02, TMP-03, MOCK-01, MOCK-02, MOCK-03, MOCK-04
**Success Criteria** (what must be TRUE):

  1. L'utente può aprire l'app su macOS, Linux e Windows, vedere il layout a due pannelli ridimensionabile con tema Yaru (chiaro/scuro/sistema), e il toggle tema funziona senza riavviare l'app
  2. L'utente può compilare tutti i 15 campi del form (inclusi dropdown contestuale pesi, campo password token con toggle visibilità, seed con bottone dado, e warning upscaler/cache) e premere Genera — tutti i campi si disabilitano, la barra di avanzamento avanza con contatore "Step N / totale", e si riabilita al termine
  3. Al termine della generazione mock (~5 secondi), il pannello destro mostra un'immagine placeholder; l'utente può premere Salva, scegliere una cartella e trovare il file PNG salvato con nome `{preset}_{seed}_{timestamp}.png`
  4. I file temporanei di sessioni precedenti (crash) vengono rimossi all'avvio; i file della sessione corrente vengono rimossi alla chiusura normale dell'app
  5. La scorciatoia Cmd/Ctrl+Enter avvia la generazione esattamente come il bottone Genera

**Plans:** 3/3 plans complete

Plans:

- [x] 01-01-PLAN.md -- Walking skeleton: Flutter project scaffold, two-panel Yaru layout, mock generation service, progress bar, placeholder image
- [x] 01-02-PLAN.md -- Complete form: all 15 CLI fields in 4 collapsible sections, preset catalog, field validation, keyboard shortcut
- [x] 01-03-PLAN.md -- Output panel: save flow with file_picker, temp directory lifecycle management

**UI hint**: yes

### Phase 2: Rust Bridge Wiring

**Goal**: L'utente può avviare una vera generazione di immagini con diffusion-rs direttamente dalla GUI, con preview live aggiornata ad ogni step e immagine finale reale — nessun mock.
**Mode**: mvp
**Depends on**: Phase 1
**Requirements**: FRB-01, FRB-02, FRB-03, FRB-04, FRB-05, FRB-06, FRB-07, FRB-08, FRB-09
**Success Criteria** (what must be TRUE):

  1. Il dropdown preset nella GUI è popolato dinamicamente da `get_presets()` Rust (non da lista hardcoded Dart); il dropdown pesi si aggiorna contestualmente via `get_weights_for_preset()`
  2. Premendo Genera con parametri validi, il pannello destro mostra preview live aggiornate ad ogni step di diffusione, e al termine compare l'immagine finale generata da diffusion-rs
  3. Un panic Rust durante la generazione non causa crash della GUI: l'errore è intercettato da `catch_unwind`, la UI si riabilita e mostra un messaggio di errore leggibile
  4. La CI verifica automaticamente che i file generati da FRB codegen siano sincronizzati con il codebase Rust (diff check fallisce la build se desincronizzati)

**Plans:** 2/2 plans complete

Plans:

- [x] 02-01-PLAN.md -- Rust crate scaffold: gui/rust/ with GuiParams DTO, get_presets(), get_weights_for_preset(), generate_image_stream(), catch_unwind, Progress pub fields
- [x] 02-02-PLAN.md -- Dart integration: FRB codegen, RustGenerationService, provider swap, error dialog, output panel downloading state + live preview

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Flutter UI Foundation (Mock Mode) | 3/3 | Complete   | 2026-06-18 |
| 2. Rust Bridge Wiring | 2/2 | Complete   | 2026-06-21 |
