import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

/// Notifier managing the application theme mode (per UI-03, UI-04, UI-05).
/// Default value is [ThemeMode.system] so the app follows the OS theme
/// preference on first launch (per UI-04).
class ThemeModeNotifier extends Notifier<ThemeMode> {
  @override
  ThemeMode build() => ThemeMode.system;

  void setThemeMode(ThemeMode mode) {
    state = mode;
  }
}

/// Provider for the current [ThemeMode] with Light/System/Dark toggle.
final themeModeProvider =
    NotifierProvider<ThemeModeNotifier, ThemeMode>(ThemeModeNotifier.new);
