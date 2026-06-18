import 'dart:io';

import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:path_provider/path_provider.dart';
import 'package:uuid/uuid.dart';

/// Manages a session-isolated temp directory for generated images.
///
/// On startup ([initialize]):
///   - Cleans stale session directories from previous crashes (TMP-03)
///   - Creates a new uniquely-named session directory (TMP-01)
///
/// On shutdown ([cleanup]):
///   - Deletes the current session directory (TMP-02)
///
/// The singleton pattern ensures a single session directory per app run.
/// A Riverpod provider ([tempDirectoryManagerProvider]) exposes the
/// instance for consistency with the rest of the app (D-09).
class TempDirectoryManager {
  TempDirectoryManager._();

  /// Singleton instance.
  static final TempDirectoryManager instance = TempDirectoryManager._();

  /// Prefix identifying session directories owned by this app.
  static const String prefix = 'diffusion_rs_gui_';

  late Directory _sessionDir;

  /// Absolute path to the current session's temp directory.
  String get sessionPath => _sessionDir.path;

  /// Initialises the session directory.
  ///
  /// Must be called once at app startup (before [runApp]).
  /// 1. Resolves the platform temp root via path_provider.
  /// 2. Removes stale session directories left by previous crashes (TMP-03).
  /// 3. Creates a fresh session directory named `{prefix}{uuid_v4}` (TMP-01).
  Future<void> initialize() async {
    final tempRoot = await getTemporaryDirectory();
    await _cleanStaleSessionDirs(tempRoot);

    final sessionId = const Uuid().v4();
    _sessionDir = Directory('${tempRoot.path}/$prefix$sessionId');
    await _sessionDir.create(recursive: true);
  }

  /// Best-effort removal of directories from previous sessions.
  ///
  /// Iterates all entries in [tempRoot], filtering for directories whose
  /// name starts with [prefix]. Each deletion is wrapped in try/catch so
  /// a single permission error (common on Windows -- see RESEARCH.md
  /// Pitfall 5) does not abort the loop.
  Future<void> _cleanStaleSessionDirs(Directory tempRoot) async {
    try {
      await for (final entity in tempRoot.list()) {
        if (entity is Directory) {
          final dirName = entity.uri.pathSegments
              .lastWhere((s) => s.isNotEmpty, orElse: () => '');
          if (dirName.startsWith(prefix)) {
            try {
              await entity.delete(recursive: true);
            } catch (_) {
              // Best-effort: skip directories we cannot delete.
            }
          }
        }
      }
    } catch (_) {
      // Best-effort: if we cannot list the temp root, proceed without
      // cleaning stale directories. The app can still create its own.
    }
  }

  /// Deletes the current session directory (TMP-02).
  ///
  /// Called on normal app exit. Wrapped in try/catch for best-effort
  /// cleanup -- if the directory is already gone or locked, the failure
  /// is silently ignored.
  Future<void> cleanup() async {
    try {
      if (await _sessionDir.exists()) {
        await _sessionDir.delete(recursive: true);
      }
    } catch (_) {
      // Best-effort cleanup.
    }
  }
}

/// Riverpod provider exposing the [TempDirectoryManager] singleton.
///
/// Consistent with D-09: all cross-feature state is accessed via providers.
final tempDirectoryManagerProvider = Provider<TempDirectoryManager>((ref) {
  return TempDirectoryManager.instance;
});
