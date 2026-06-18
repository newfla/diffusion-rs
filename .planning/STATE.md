---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
current_phase: 01
current_phase_name: flutter-ui-foundation-mock-mode
status: executing
stopped_at: Phase 1 UI-SPEC approved
last_updated: "2026-06-18T15:18:27.498Z"
progress:
  total_phases: 2
  completed_phases: 0
  total_plans: 3
  completed_plans: 2
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-18)

**Core value:** L'utente può configurare e avviare una generazione di immagini con lo stesso set di opzioni della CLI, senza aprire un terminale.
**Current focus:** Phase 01 — flutter-ui-foundation-mock-mode

## Current Position

**Phase:** 01 (flutter-ui-foundation-mock-mode) — EXECUTING
**Plan:** 3 of 3
**Status:** Ready to execute
**Progress:** [███████░░░] 67%

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

Last session: 2026-06-18T15:18:27.485Z
Stopped at: Phase 1 UI-SPEC approved

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
