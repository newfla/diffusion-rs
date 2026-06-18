---
phase: 01-flutter-ui-foundation-mock-mode
plan: 03
subsystem: ui
tags: [flutter, yaru, riverpod, file-picker, path-provider, temp-directory, save-flow]

requires:
  - phase: 01-01
    provides: Walking skeleton with two-panel layout, GenerationNotifier, output panel 4-state machine
  - phase: 01-02
    provides: ParamsNotifier with presetName and seed, complete form fields
provides:
  - TempDirectoryManager singleton with session-isolated temp directory lifecycle (create, stale cleanup, exit cleanup)
  - OutputNotifier with saveImage() using OS-native file picker, correct filename format, SnackBar confirmation
  - Complete 5-state output panel (idle, spinner, progress, complete+save, error)
  - GenerationNotifier writes placeholder to session temp dir on completion
affects: [02-rust-ffi-bridge]

tech-stack:
  added: []
  patterns: [TempDirectoryManager singleton with Riverpod provider wrapper, AppLifecycleListener.onExitRequested for desktop cleanup, FilePicker.saveFile static API (v11), Platform.environment for Pictures directory resolution]

key-files:
  created:
    - gui/lib/shared/services/temp_directory_manager.dart
    - gui/lib/features/output/providers/output_provider.dart
  modified:
    - gui/lib/main.dart
    - gui/lib/features/output/output_panel.dart
    - gui/lib/features/generation/providers/generation_provider.dart

key-decisions:
  - "file_picker v11 uses static FilePicker.saveFile() not FilePicker.platform.saveFile() -- adapted API call"
  - "AppLifecycleListener.onExitRequested used for desktop cleanup instead of WidgetsBindingObserver -- provides cleaner exit hook with AppExitResponse"
  - "TempDirectoryManager exposed as both singleton and Riverpod provider for flexibility -- singleton for main.dart init, provider for cross-feature access"

patterns-established:
  - "TempDirectoryManager.instance.initialize() in main.dart before runApp -- ensures temp dir ready for all widgets"
  - "OutputNotifier.saveImage() pattern: resolve Pictures dir, show save dialog, copy file, show SnackBar -- reusable for Phase 2"
  - "AppLifecycleListener.onExitRequested for desktop app cleanup hooks"

requirements-completed: [OUT-01, OUT-02, OUT-03, OUT-04, OUT-05, OUT-06, TMP-01, TMP-02, TMP-03]

duration: 7min
completed: 2026-06-18
status: complete
---

# Phase 01 Plan 03: Output Panel and Temp Directory Summary

**Complete output panel with OS-native save dialog (file_picker v11), session-isolated temp directory lifecycle, and SnackBar confirmation -- all generated images written to UUID-named temp dir, cleaned on startup and exit**

## Performance

- **Duration:** 7 min
- **Started:** 2026-06-18T15:23:01Z
- **Completed:** 2026-06-18T15:30:37Z
- **Tasks:** 2
- **Files modified:** 5 (2 created, 3 modified)

## Accomplishments

- Built TempDirectoryManager with session-isolated temp directory: creates `diffusion_rs_gui_{uuid}` on startup, cleans stale sessions from previous crashes, cleans current session on app exit via AppLifecycleListener
- Created OutputNotifier with saveImage() that opens OS-native file picker, defaults to Pictures directory, saves with `{preset}_{seed}_{timestamp}.png` filename format, and shows SnackBar "Saved to {path}" for 4 seconds
- Completed the output panel 5-state machine: idle (icon + text), spinner (pre-progress), progress bar + step counter, complete (image + Save button), error
- Updated GenerationNotifier to write placeholder image to session temp dir instead of raw temp directory

## Task Commits

Each task was committed atomically:

1. **Task 1: Temp directory manager with session isolation and lifecycle cleanup** - `1a9c265` (feat)
2. **Task 2: Output panel complete state machine with save flow and SnackBar** - `e5dc9e1` (feat)

## Files Created/Modified

- `gui/lib/shared/services/temp_directory_manager.dart` - Singleton managing session temp dir lifecycle: initialize (clean stale + create), cleanup, sessionPath getter, Riverpod provider
- `gui/lib/features/output/providers/output_provider.dart` - OutputState + OutputNotifier with saveImage() using FilePicker.saveFile, SnackBar, try/catch for Linux compatibility
- `gui/lib/main.dart` - Added WidgetsFlutterBinding.ensureInitialized, TempDirectoryManager.initialize(), AppLifecycleListener cleanup registration
- `gui/lib/features/output/output_panel.dart` - Wired Save button to OutputNotifier.saveImage() with params from paramsProvider; complete state now functional
- `gui/lib/features/generation/providers/generation_provider.dart` - Uses TempDirectoryManager.sessionPath for output file; timestamped filename for uniqueness

## Decisions Made

- **file_picker v11 static API**: file_picker v11.0.2 removed `FilePicker.platform` accessor; methods are now static on `FilePicker` directly (e.g. `FilePicker.saveFile()`). Adapted the call site accordingly.
- **AppLifecycleListener over WidgetsBindingObserver**: Used `AppLifecycleListener.onExitRequested` which provides a clean `AppExitResponse` return value for desktop exit hooks, rather than the more complex `WidgetsBindingObserver.didChangeAppLifecycleState`.
- **Dual access pattern for TempDirectoryManager**: Exposed as singleton (`TempDirectoryManager.instance`) for use in `main()` before Riverpod is initialized, and also as a Riverpod provider (`tempDirectoryManagerProvider`) for widget/notifier access consistent with D-09.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed file_picker v11 API change**
- **Found during:** Task 2 (flutter analyze)
- **Issue:** Plan specified `FilePicker.platform.saveFile()` but file_picker v11.0.2 uses static `FilePicker.saveFile()` -- the `platform` getter no longer exists
- **Fix:** Changed call from `FilePicker.platform.saveFile(...)` to `FilePicker.saveFile(...)`
- **Files modified:** gui/lib/features/output/providers/output_provider.dart
- **Verification:** `flutter analyze` passes with no issues
- **Committed in:** e5dc9e1 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Necessary API adaptation for file_picker v11. No scope creep.

## Issues Encountered

None beyond the auto-fixed deviation above.

## Known Stubs

None. All save flow functionality is wired end-to-end. The placeholder image from mock generation is intentional for Phase 1 and will be replaced by real generated images in Phase 2.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 1 complete: all 3 plans executed, all requirements (SETUP, UI, FORM, GEN, OUT, TMP, MOCK) satisfied
- GenerationService seam ready for Phase 2 Rust FFI swap -- single provider line change
- TempDirectoryManager ready for Phase 2: real generated images will be written to the same session temp dir
- OutputNotifier.saveImage() works with any source file path -- no changes needed when real images replace placeholder

## Self-Check: PASSED

All 5 files (2 created, 3 modified) verified on disk. Both task commits (1a9c265, e5dc9e1) verified in git log.

---
*Phase: 01-flutter-ui-foundation-mock-mode*
*Plan: 03*
*Completed: 2026-06-18*
