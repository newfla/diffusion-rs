---
status: complete
phase: 01-flutter-ui-foundation-mock-mode
source: [01-01-SUMMARY.md, 01-02-SUMMARY.md, 01-03-SUMMARY.md]
started: 2026-06-18T18:00:00Z
updated: 2026-06-21T00:00:00Z
---

## Current Test

[testing complete]

## Tests

### 1. App launches (two-panel layout)
expected: |
  Avvia l'app con `flutter run -d macos` dalla cartella `gui/`.
  L'app si apre senza errori, mostra due pannelli affiancati separati da un divisore trascinabile.
  Il pannello sinistro contiene la form; quello destro mostra lo stato idle (icona + testo "No image yet" o simile).
result: pass

### 2. Theme toggle
expected: |
  In alto a destra è visibile un SegmentedButton con tre opzioni (Light / System / Dark).
  Cliccando su Light: tema chiaro applicato.
  Cliccando su Dark: tema scuro applicato.
  Cliccando su System: tema segue le preferenze di sistema.
result: pass

### 3. Preset dropdown e auto-populate defaults
expected: |
  Nella sezione Model, il dropdown Preset mostra tutti i 41 preset.
  Selezionando StableDiffusion1_5: i campi Steps si imposta a 20, Width a 512, Height a 512.
  Selezionando Flux1Dev: Steps → 28, Width → 1024, Height → 1024.
  Selezionando StableDiffusion3_5Large: Steps → 28, Width → 1024, Height → 1024.
result: pass

### 4. Weights dropdown
expected: |
  Per preset senza varianti (es. StableDiffusion1_4): dropdown Weights mostra "N/A" ed è disabilitato.
  Per preset con varianti (es. Flux1Dev): dropdown mostra i pesi disponibili (Q2_K, Q3_K, Q4_0, Q4_K, Q8_0) e il default è Q2_K.
  Cambiando preset, il dropdown Weights si aggiorna automaticamente.
result: pass

### 5. Form fields — prompt e campi opzionali
expected: |
  Il campo Prompt è multilinea (almeno 3 righe) e obbligatorio.
  I campi Steps, Width, Height mostrano i valori di default del preset selezionato.
  Il bottone dado (dice) accanto a Seed genera un nuovo valore casuale nel campo Seed.
  Negative prompt è presente e accetta testo.
result: issue
reported: "Il campo dado non genera alcun valore"
severity: major

### 6. FORM-15 warning
expected: |
  Nella sezione Advanced, seleziona un upscaler qualsiasi (es. RealESRGAN_x4plus) e lascia Cache mode = None.
  Appare subito un testo di avviso rosso: "Upscaler is active without caching. Select a cache mode to avoid recomputing all steps during upscaling." (o testo simile).
  Selezionando un cache mode (es. UCACHE), il warning scompare.
result: pass

### 7. Generate flow: idle → spinner → progress → complete
expected: |
  Scrivi un testo qualsiasi nel campo Prompt.
  Premi il bottone Generate.
  
  a) Il bottone mostra "Generating..." e si disabilita.
  b) Nel pannello destro: appare uno spinner indeterminato (Yaru spinner circolare).
  c) Dopo il primo step: lo spinner lascia posto a una progress bar + testo "Step 1 / 20".
  d) La barra avanza step per step fino a Step 20 / 20.
  e) Al completamento: appare l'immagine placeholder (grigio chiaro) + bottone Save.
  f) Il bottone Generate torna abilitato.
  
  L'intera sequenza dura circa 5 secondi.
result: pass

### 8. Cmd+Enter shortcut
expected: |
  Con un prompt non vuoto scritto nella form, premi Cmd+Enter (macOS) o Ctrl+Enter (Linux/Windows).
  La generazione parte come se avessi cliccato Generate.
  Se il prompt è vuoto, la shortcut non fa nulla (bottone disabilitato).
result: pass

### 9. Save image
expected: |
  Dopo che una generazione è completata (placeholder visibile), clicca il bottone Save.
  Si apre un dialogo OS nativo per scegliere dove salvare il file.
  Il nome file suggerito ha il formato `{preset}_{seed}_{timestamp}.png`.
  Dopo aver confermato: appare una SnackBar in basso con il testo "Saved to {path}".
  Il file è effettivamente presente nel path indicato.
result: pass

### 10. Temp directory lifecycle
expected: |
  Prima di avviare l'app: apri il Finder (o terminale) e naviga in /var/folders o $TMPDIR.
  Avvia l'app: appare una cartella `diffusion_rs_gui_{uuid}`.
  Chiudi l'app normalmente (Cmd+Q o chiudi la finestra): la cartella diffusion_rs_gui_{uuid} viene eliminata.
result: pass

## Summary

total: 10
passed: 9
issues: 1
pending: 0
skipped: 0
blocked: 0

## Gaps

- truth: "Il bottone dado (dice) accanto a Seed genera un nuovo valore casuale nel campo Seed"
  status: failed
  reason: "User reported: Il campo dado non genera alcun valore"
  severity: major
  test: 5
  root_cause: ""
  artifacts: []
  missing: []
  debug_session: ""
