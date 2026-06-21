# Phase 2: Rust Bridge Wiring - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-21
**Phase:** 2-rust-bridge-wiring
**Areas discussed:** Preview frames per step, Error & download UX, FRB codegen workflow

---

## Preview frames per step

| Option | Description | Selected |
|--------|-------------|----------|
| Solo progress bar, immagine finale alla fine | Nessuna preview intermedia. Zero modifiche al Progress struct. | |
| Immagini intermedie reali per ogni step | Preview reale per ogni step. Richiede estendere Progress o meccanismo file. | ✓ |

**User's choice:** Immagini intermedie reali per ogni step, con approccio file-based su disco.
**Notes:** L'utente ha specificato esplicitamente di usare `preview_output` + `PreviewType::PREVIEW_PROJ`, e ha indicato di guardare come la CLI gestisce `args.preview == FAST` (`cli/src/main.rs` linee 188-191).

### Race condition: preview file vs progress event

| Option | Description | Selected |
|--------|-------------|----------|
| Read-after-write nello stesso step (accetta il rischio) | Dart legge il file dopo ogni progress event, accetta frame obsoleto se race. | ✓ |
| Preview file come stato condiviso (Mutex in Rust) | Nessun race, ma richiede stato condiviso nel layer FRB. | |
| Non mi interessa, decidi tu | Claude sceglie l'approccio più pragmatico. | |

**User's choice:** Read-after-write, race accettato.
**Notes:** Se il file non è pronto, `previewImage` è null e la UI mostra il frame precedente (graceful degradation).

---

## Error & download UX

### Download modello non in cache

| Option | Description | Selected |
|--------|-------------|----------|
| Spinner + testo statico "Downloading model..." | Spinner nel pannello destro con testo fisso fino all'arrivo del primo step. | ✓ |
| Solo spinner | Nessun testo aggiuntivo. | |
| Non gestire in Phase 2 | Assume modello già presente in cache. | |

**User's choice:** Spinner + testo "Downloading model..." fino al primo `ProgressEvent`.

### Errore Rust durante la generazione

| Option | Description | Selected |
|--------|-------------|----------|
| Testo di errore nel pannello + SnackBar | Errore nel pannello destro + SnackBar. | |
| Solo SnackBar di errore | SnackBar breve, pannello resta invariato. | |
| Dialog modale di errore | AlertDialog blocca la UI fino a OK. | ✓ |

**User's choice:** Dialog modale.
**Notes:** "Generation Failed" come titolo, testo Rust come body, singolo pulsante OK.

---

## FRB codegen workflow

### Esecuzione codegen

| Option | Description | Selected |
|--------|-------------|----------|
| Script Makefile + check CI | `make codegen` per sviluppatori, CI fa `git diff --exit-code`. | |
| Script shell standalone | `gui/scripts/codegen.sh` + CI diff check. | |
| Integrato nel build Flutter | Codegen automatico a ogni `flutter build`. Nessun passo manuale. | ✓ |

**User's choice:** Integrato nel build Flutter.

### CI diff check (FRB-08)

| Option | Description | Selected |
|--------|-------------|----------|
| CI esegue codegen + git diff | Job CI verifica sync dei file generati. | |
| Non fare il check in CI | Fidarsi del build Flutter integrato. FRB-08 non implementato. | ✓ |

**User's choice:** Nessun CI diff check — build integrato garantisce sync. FRB-08 waivato per Phase 2.

---

## Claude's Discretion

- Exact FRB 2.x annotation syntax e forma dello streaming API (`StreamSink` vs callback)
- Timeout valore per rilevare stato "Downloading model..." in Dart
- `preview_interval` value (default 1)
- Come integrare codegen nel build Flutter (hook specifico di FRB 2.x)

## Deferred Ideas

- **FRB-08 CI diff check** — waivato, può essere rivisto se il build integrato non è affidabile
- **Download progress (MDL-01)** — v2
- **Generation cancellation (UX-03)** — v2, richiede segnale abort nel C++
- **In-memory preview bytes** — alternativa al file-based scartata per complessità
