---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
current_phase: 02
current_phase_name: rust-bridge-wiring
status: executing
stopped_at: Phase 2 UI-SPEC approved
last_updated: "2026-06-21T17:37:11.138Z"
progress:
  total_phases: 2
  completed_phases: 1
  total_plans: 5
  completed_plans: 3
  percent: 50
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-18)

**Core value:** L'utente può configurare e avviare una generazione di immagini con lo stesso set di opzioni della CLI, senza aprire un terminale.
**Current focus:** Phase 02 — rust-bridge-wiring

## Current Position

**Phase:** 02 (rust-bridge-wiring) — EXECUTING
**Plan:** 1 of 2
**Status:** Executing Phase 02
**Progress:** [██████████] 100%

## Performance Metrics

**Phases complete:** 0/2
**Plans complete:** 0/?
**Requirements covered:** 46/46

## Accumulated Context

### Key Decisions

- flutter_rust_bridge 2.x per FFI Dart↔Rust — unica soluzione matura per desktop
- Phase 1 mock-first per disaccoppiare sviluppo UI da build Rust/GPU
- Yaru 6.x come design system — light/dark built-in, YaruPasswordField per token
- Riverpod 2.x (AsyncNotifier) per state management — 4 provider: params, generation lifecycle, progress, theme
- gui/rust/ NON membro del workspace root Cargo.toml — evita trigger build CMake/GPU

### Critical Pre-requisites (Phase 2)

- SETUP-03: token.txt placeholder da committare subito (sblocca CI fresh checkout)
- FRB-05: campi `step`, `steps`, `time` in `src/api.rs` Progress struct devono diventare `pub`
- SETUP-02: gui/rust/ come workspace Cargo isolato (non membro root workspace)

### Blockers

None

### Todos

- [ ] Plan Phase 1 (`/gsd-plan-phase 1`)

## Session Continuity

**Resume file:** .planning/phases/02-rust-bridge-wiring/02-UI-SPEC.md

Last session: 2026-06-21T17:03:28.456Z
Stopped at: Phase 2 UI-SPEC approved

## Performance Metrics

| Phase | Plan | Duration | Notes |
|-------|------|----------|-------|
| Phase 01 P01 | 24min | 2 tasks | 68 files |
| Phase 01 P02 | 24min | 2 tasks | 9 files |

## Decisions

- [Phase ?]: MultiSplitView v3.6.2 uses builder callback, not children property
- [Phase ?]: Root .gitignore *.png overridden via gui/.gitignore negation for Flutter assets
- [Phase ?]: Used Notifier<GenerationState> (sync) with async generate() method for generation lifecycle
- [Phase ?]: Used DropdownButton+InputDecorator instead of deprecated DropdownButtonFormField.value in Flutter 3.44.x
- [Phase ?]: Preset catalog has 41 presets (verified against src/preset.rs Preset enum)
