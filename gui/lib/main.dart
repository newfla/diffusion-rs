import 'dart:ui';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'app.dart';
import 'shared/services/temp_directory_manager.dart';
import 'src/rust/frb_generated.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Initialize flutter_rust_bridge runtime before any FFI calls.
  await RustLib.init();

  // TMP-01/TMP-03: create session temp dir and clean stale dirs from
  // previous crashes before the app starts.
  await TempDirectoryManager.instance.initialize();

  // TMP-02: register cleanup on normal app exit.
  // AppLifecycleListener fires on desktop when the window is closed.
  AppLifecycleListener(
    onExitRequested: () async {
      await TempDirectoryManager.instance.cleanup();
      return AppExitResponse.exit;
    },
  );

  runApp(
    const ProviderScope(
      child: DiffusionRsApp(),
    ),
  );
}
