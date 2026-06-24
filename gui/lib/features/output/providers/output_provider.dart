import 'dart:io';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

/// Immutable state for the output/save lifecycle.
class OutputState {
  /// Path of the last successfully saved file.
  /// Used to confirm save completion; the UI shows a SnackBar with this path.
  final String? lastSavedPath;

  const OutputState({this.lastSavedPath});

  OutputState copyWith({String? lastSavedPath}) {
    return OutputState(lastSavedPath: lastSavedPath ?? this.lastSavedPath);
  }
}

/// Riverpod Notifier managing the output/save lifecycle.
///
/// Provides [saveImage] which opens the OS-native file picker (OUT-04),
/// copies the generated image to the chosen destination (OUT-05),
/// and shows a SnackBar confirmation (D-14).
class OutputNotifier extends Notifier<OutputState> {
  @override
  OutputState build() => const OutputState();

  /// Opens the OS-native save dialog and copies the generated image.
  ///
  /// [sourcePath] - path to the temp file produced by the generation service.
  /// [presetName] - current preset name for the default filename (OUT-05).
  /// [seed]       - current seed value for the default filename (OUT-05).
  /// [context]    - BuildContext for ScaffoldMessenger SnackBar (D-14).
  ///
  /// The default filename format is `{preset}_{seed}_{timestamp}.png`.
  /// The initial directory targets the system Pictures folder (OUT-06).
  /// Wraps the file_picker call in try/catch (RESEARCH.md Pitfall 4:
  /// Linux may lack zenity/kdialog).
  Future<void> saveImage(
    String sourcePath,
    String presetName,
    int seed,
    BuildContext context,
  ) async {
    final timestamp = DateTime.now().millisecondsSinceEpoch;
    final defaultFilename = '${presetName}_${seed}_$timestamp.png';

    // OUT-06: attempt to resolve the system Pictures directory.
    String? initialDirectory;
    try {
      final home = Platform.environment['HOME'] ??
          Platform.environment['USERPROFILE'];
      if (home != null) {
        final picturesDir = Directory('$home/Pictures');
        if (await picturesDir.exists()) {
          initialDirectory = picturesDir.path;
        }
      }
    } catch (_) {
      // Fall through to file_picker default if we cannot resolve Pictures.
    }

    try {
      final outputPath = await FilePicker.saveFile(
        dialogTitle: 'Save image',
        fileName: defaultFilename,
        initialDirectory: initialDirectory,
      );

      if (outputPath != null && context.mounted) {
        // Copy the temp file to the user-chosen location.
        final sourceFile = File(sourcePath);
        await sourceFile.copy(outputPath);

        state = OutputState(lastSavedPath: outputPath);

        // D-14: show confirmation SnackBar for 4 seconds.
        if (context.mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text('Saved to $outputPath'),
              duration: const Duration(seconds: 4),
            ),
          );
        }
      }
    } catch (e) {
      // Pitfall 4: file_picker may throw PlatformException on Linux
      // if zenity/kdialog is missing.
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Failed to save image: $e'),
            duration: const Duration(seconds: 4),
          ),
        );
      }
    }
  }
}

/// Provider for the output/save lifecycle state.
final outputProvider = NotifierProvider<OutputNotifier, OutputState>(
  OutputNotifier.new,
);
