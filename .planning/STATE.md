---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
current_phase: 1
status: planning
stopped_at: Phase 1 UI-SPEC approved
last_updated: "2026-06-18T13:10:48.839Z"
progress:
  total_phases: 2
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-18)

**Core value:** L'utente può configurare e avviare una generazione di immagini con lo stesso set di opzioni della CLI, senza aprire un terminale.
**Current focus:** Phase 1 — Flutter UI Foundation (Mock Mode)

## Current Position

**Phase:** 1 of 2
**Plan:** None (not yet planned)
**Status:** Ready to plan
**Progress:** ░░░░░░░░░░ 0%

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

**Resume file:** .planning/phases/01-flutter-ui-foundation-mock-mode/01-UI-SPEC.md

Last session: 2026-06-18T13:10:48.834Z
Stopped at: Phase 1 UI-SPEC approved
