---
status: complete
phase: 02-rust-bridge-wiring
source: [02-01-SUMMARY.md, 02-02-SUMMARY.md]
started: 2026-06-23T14:40:00Z
updated: 2026-06-23T14:40:00Z
---

## Current Test

## Current Test

[testing complete]

## Tests

### 1. Cold Start — app si avvia senza crash
expected: |
  Chiudi completamente l'app se aperta.
  Lancia l'app con `flutter run -d macos` oppure apri il .app dalla cartella build.
  L'app si apre, mostra i due pannelli (nessuna schermata nera), e il dropdown Preset
  è già popolato con i preset reali (almeno 10 voci).
  Nessun crash, nessun dialog di errore all'avvio.
result: pass

### 2. Preset e Weights da Rust FFI
expected: |
  Nel dropdown Preset seleziona "Flux1Dev".
  Il dropdown Weights si aggiorna e mostra le varianti reali (Q2_K, Q3_K, Q4_0, Q4_K, Q8_0).
  Seleziona un preset senza pesi (es. StableDiffusion1_4): il dropdown Weights mostra "N/A" e si disabilita.
  Questi dati provengono dal codice Rust (get_presets / get_weights_for_preset via FFI), non da mock.
result: pass

### 3. Avvio generazione reale — stato "Downloading model..."
expected: |
  Seleziona un preset leggero (es. StableDiffusion1_5, Q4_0 o simile).
  Scrivi un prompt breve (es. "a red apple").
  Premi Generate.
  
  Prima che l'inferenza inizi: nel pannello destro appare uno spinner con testo
  "Downloading model..." (o simile) durante il download del modello da HuggingFace.
  Il bottone Generate è disabilitato e mostra "Generating...".
  
  Nota: il download può richiedere diversi minuti alla prima esecuzione.
result: pass

### 4. Live preview durante l'inferenza
expected: |
  Dopo il download del modello, l'inferenza parte.
  Nel pannello destro: la progress bar avanza step per step.
  Ad ogni step (o ogni N step) appare un'immagine di anteprima live che si aggiorna
  man mano che la generazione procede.
  Non è un placeholder grigio — è la preview reale parzialmente denoised.
result: pass

### 5. Immagine finale reale
expected: |
  Al completamento: il pannello destro mostra l'immagine finale generata da diffusion-rs.
  L'immagine è una foto/illustrazione coerente col prompt (non un placeholder grigio).
  Il bottone Generate torna abilitato.
  Il bottone Save appare.
result: pass

### 6. Error dialog su generazione fallita
expected: |
  (Testa questo solo se hai modo di provocare un errore — es. seleziona un preset con
  un path di modello inesistente, oppure osserva se viene mostrato se una generazione
  precedente fallisce.)
  
  Se la generazione fallisce: appare un AlertDialog con titolo "Generation Failed"
  e un messaggio d'errore. Il bottone OK chiude il dialogo.
  L'app non crasha — torna allo stato idle.
result: skipped
reason: non testabile senza configurazione ad hoc per provocare un errore

## Summary

total: 6
passed: 5
issues: 0
pending: 0
skipped: 1
blocked: 0

## Gaps

[none yet]
