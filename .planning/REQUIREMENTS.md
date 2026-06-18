# Requirements: diffusion-rs GUI

**Defined:** 2026-06-18
**Core Value:** L'utente può configurare e avviare una generazione di immagini con lo stesso set di opzioni della CLI, senza aprire un terminale.

## v1 Requirements

### Setup & Struttura Progetto

- [x] **SETUP-01**: Il progetto Flutter risiede in `gui/` come sottocartella del monorepo diffusion-rs esistente
- [x] **SETUP-02**: Il bridge crate Rust risiede in `gui/rust/` come workspace Cargo isolato (non membro del root workspace `Cargo.toml`)
- [x] **SETUP-03**: Un placeholder `token.txt` vuoto è committato nella root del repo per sbloccare le build CI
- [x] **SETUP-04**: La app Flutter compila ed esegue su macOS, Linux e Windows senza modifiche al codice

### UI Layout

- [x] **UI-01**: L'interfaccia è divisa in due pannelli affiancati: sinistra (form parametri) e destra (preview + output)
- [x] **UI-02**: I pannelli sono ridimensionabili tramite drag handle orizzontale
- [x] **UI-03**: La UI supporta tema chiaro e scuro con il design system Yaru
- [x] **UI-04**: Il tema segue le impostazioni di sistema per default
- [x] **UI-05**: L'utente può sovrascrivere il tema manualmente tramite un toggle (Chiaro / Sistema / Scuro)

### Form Parametri (Pannello Sinistro)

- [ ] **FORM-01**: Dropdown per selezione preset (lista di tutti i `PresetDiscriminants` disponibili)
- [ ] **FORM-02**: Dropdown per selezione pesi (visibile solo se il preset selezionato supporta varianti di peso; le opzioni cambiano contestualmente al preset)
- [ ] **FORM-03**: Campo testo multiline per il prompt di generazione (obbligatorio)
- [ ] **FORM-04**: Campo testo per il negative prompt (opzionale)
- [ ] **FORM-05**: Campo numerico per il numero di inference steps (opzionale, override del default del preset)
- [ ] **FORM-06**: Campi numerici per larghezza e altezza output in pixel (opzionali, override del default)
- [ ] **FORM-07**: Campo numerico per il numero di immagini da generare in batch (default: 1)
- [ ] **FORM-08**: Campo numerico per il seed RNG con bottone dado che azzera il valore a -1 (random)
- [ ] **FORM-09**: Dropdown per la modalità di caching (Nessuno / UCACHE / EASYCACHE / DBCACHE / TAYLORSEER / CACHEDIT / SPECTRUM)
- [ ] **FORM-10**: Dropdown per la preview durante la generazione (Nessuna / Fast / Accurate)
- [ ] **FORM-11**: Dropdown per la modalità upscaler (Nessuno / 8 modalità disponibili)
- [ ] **FORM-12**: Campo numerico per il fattore di scala upscaler (visibile solo se upscaler ≠ Nessuno; default: 2.0)
- [ ] **FORM-13**: Campo token HuggingFace come campo password (testo oscurato, bottone toggle visibilità)
- [ ] **FORM-14**: Toggle per la modalità low VRAM (VAE tiling + flash attention)
- [ ] **FORM-15**: Warning inline visibile quando upscaler è selezionato ma cache è "Nessuno" (o auto-selezione default cache)

### Controlli Generazione

- [x] **GEN-01**: Bottone "Genera" che avvia la generazione
- [ ] **GEN-02**: Alla pressione di "Genera", tutti i campi del form vengono disabilitati per tutta la durata della generazione
- [x] **GEN-03**: Barra di avanzamento lineare visibile durante la generazione
- [x] **GEN-04**: Contatore di step testuale accanto alla barra ("Step N / totale")
- [ ] **GEN-05**: Al completamento della generazione, tutti i campi del form vengono riabilitati
- [ ] **GEN-06**: Scorciatoia da tastiera Cmd/Ctrl+Enter equivalente al bottone Genera

### Pannello Destro — Preview & Output

- [ ] **OUT-01**: Il pannello destro mostra la preview intermedia durante la generazione (aggiornata ad ogni evento progress)
- [ ] **OUT-02**: Al completamento della generazione, il pannello mostra l'immagine finale
- [ ] **OUT-03**: L'immagine preview/finale occupa lo spazio disponibile mantenendo il rapporto d'aspetto
- [ ] **OUT-04**: Bottone "Salva" visibile dopo il completamento della generazione
- [ ] **OUT-05**: La pressione di "Salva" apre un folder picker; il file viene salvato come PNG con nome `{preset}_{seed}_{timestamp}.png`
- [ ] **OUT-06**: La cartella di default per il salvataggio è la cartella Immagini/Pictures del sistema

### Gestione File Temporanei

- [ ] **TMP-01**: Tutti i file temporanei (preview PNG e output PNG) sono scritti in una directory temporanea con session ID unico
- [ ] **TMP-02**: La directory temporanea viene eliminata alla chiusura normale dell'app
- [ ] **TMP-03**: Le directory temporanee di sessioni precedenti (crash) vengono rimosse all'avvio della nuova sessione

### Mock Mode (Phase 1 — nessuna dipendenza Rust)

- [x] **MOCK-01**: In Phase 1, l'app usa `MockGenerationService`: la pressione di "Genera" avvia una sequenza di progress eventi simulati via Stream (non Timer.periodic)
- [x] **MOCK-02**: Il mock completa la "generazione" in ~5 secondi con progress step realistici
- [x] **MOCK-03**: Al termine del mock, il pannello destro mostra un'immagine placeholder predefinita
- [ ] **MOCK-04**: La lista preset e pesi in Phase 1 è hardcoded in Dart (derivata da `src/preset.rs` al momento del build)

### Bridge Rust / Wiring (Phase 2)

- [ ] **FRB-01**: `gui/rust/` espone `get_presets() → Vec<String>` via flutter_rust_bridge
- [ ] **FRB-02**: `gui/rust/` espone `get_weights_for_preset(preset: String) → Vec<String>` via flutter_rust_bridge
- [ ] **FRB-03**: `gui/rust/` espone `generate_image_stream(params: GuiParams, sink: StreamSink<ProgressEvent>)` via flutter_rust_bridge
- [ ] **FRB-04**: `GuiParams` è un DTO frb-compatibile (solo `String`, `i32`, `i64`, `f32`, `bool`, `Option<T>`) che replica tutti i 15 parametri CLI
- [ ] **FRB-05**: I campi `step`, `steps`, `time` della struct `Progress` in `src/api.rs` hanno visibilità `pub`
- [ ] **FRB-06**: Tutti gli entry point FFI in `gui/rust/` hanno wrapper `catch_unwind`
- [ ] **FRB-07**: Il profilo di build release usa `panic = "abort"` nel `gui/rust/Cargo.toml`
- [ ] **FRB-08**: La CI verifica che i file generati da FRB codegen siano aggiornati (diff check)
- [ ] **FRB-09**: `RustGenerationService` sostituisce `MockGenerationService` con una singola riga nel provider

## v2 Requirements

### UX Avanzata

- **UX-01**: History prompt con recall degli ultimi N prompt usati
- **UX-02**: Gallery output — pannello che mostra le immagini generate nella sessione corrente
- **UX-03**: Cancellazione generazione in corso (richiede segnale abort nel backend C++)
- **UX-04**: Metadata embedding (parametri di generazione) nel PNG salvato (EXIF/PNG chunk)
- **UX-05**: Lista preset raggruppata per famiglia con ricerca (rilevante a 50+ preset)

### Gestione Modelli

- **MDL-01**: UI per il download dei modelli da HuggingFace (con progress)
- **MDL-02**: UI per la cancellazione dei modelli scaricati
- **MDL-03**: Indicazione della dimensione su disco per ogni preset

## Out of Scope

| Feature | Motivo |
|---------|--------|
| Mobile (iOS/Android) | Desktop only; non compatibile con flutter_rust_bridge su mobile |
| Web | Non compatibile con FFI nativa e file system access |
| Image-to-image / img2img | Non esposto dalla CLI attuale di diffusion-rs |
| ControlNet UI | Richiede UI specializzata; non nella CLI base |
| LoRA UI | Richiede UI specializzata (file picker, strength slider); non nella CLI base |
| Generazioni multiple concorrenti | Backend single-threaded per design |
| Modelli custom (path locale) | Solo preset predefiniti nella v1 |

## Traceability

| Requisito | Fase | Stato |
|-----------|------|-------|
| SETUP-01 | Phase 1 | Complete |
| SETUP-02 | Phase 1 | Complete |
| SETUP-03 | Phase 1 | Complete |
| SETUP-04 | Phase 1 | Complete |
| UI-01 | Phase 1 | Complete |
| UI-02 | Phase 1 | Complete |
| UI-03 | Phase 1 | Complete |
| UI-04 | Phase 1 | Complete |
| UI-05 | Phase 1 | Complete |
| FORM-01 | Phase 1 | Pending |
| FORM-02 | Phase 1 | Pending |
| FORM-03 | Phase 1 | Pending |
| FORM-04 | Phase 1 | Pending |
| FORM-05 | Phase 1 | Pending |
| FORM-06 | Phase 1 | Pending |
| FORM-07 | Phase 1 | Pending |
| FORM-08 | Phase 1 | Pending |
| FORM-09 | Phase 1 | Pending |
| FORM-10 | Phase 1 | Pending |
| FORM-11 | Phase 1 | Pending |
| FORM-12 | Phase 1 | Pending |
| FORM-13 | Phase 1 | Pending |
| FORM-14 | Phase 1 | Pending |
| FORM-15 | Phase 1 | Pending |
| GEN-01 | Phase 1 | Complete |
| GEN-02 | Phase 1 | Pending |
| GEN-03 | Phase 1 | Complete |
| GEN-04 | Phase 1 | Complete |
| GEN-05 | Phase 1 | Pending |
| GEN-06 | Phase 1 | Pending |
| OUT-01 | Phase 1 | Pending |
| OUT-02 | Phase 1 | Pending |
| OUT-03 | Phase 1 | Pending |
| OUT-04 | Phase 1 | Pending |
| OUT-05 | Phase 1 | Pending |
| OUT-06 | Phase 1 | Pending |
| TMP-01 | Phase 1 | Pending |
| TMP-02 | Phase 1 | Pending |
| TMP-03 | Phase 1 | Pending |
| MOCK-01 | Phase 1 | Complete |
| MOCK-02 | Phase 1 | Complete |
| MOCK-03 | Phase 1 | Complete |
| MOCK-04 | Phase 1 | Pending |
| FRB-01 | Phase 2 | Pending |
| FRB-02 | Phase 2 | Pending |
| FRB-03 | Phase 2 | Pending |
| FRB-04 | Phase 2 | Pending |
| FRB-05 | Phase 2 | Pending |
| FRB-06 | Phase 2 | Pending |
| FRB-07 | Phase 2 | Pending |
| FRB-08 | Phase 2 | Pending |
| FRB-09 | Phase 2 | Pending |

**Coverage:**

- v1 requirements: 46 totali
- Mappati a fasi: 46/46
- Non mappati: 0 ✓

---
*Requirements defined: 2026-06-18*
*Last updated: 2026-06-18 after roadmap creation — traceability expanded to per-requirement rows*
