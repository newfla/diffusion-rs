import 'package:flutter/material.dart';

/// Shows a modal AlertDialog for generation errors (per D-05, D-06).
///
/// Title: "Generation Failed". Body: the raw Rust error string passed as
/// [errorMessage]. Single action button: TextButton with text "OK" that
/// pops the dialog. The dialog is not dismissible by tapping outside
/// (barrierDismissible: false) so the user must acknowledge via OK.
Future<void> showErrorDialog(BuildContext context, String errorMessage) {
  return showDialog<void>(
    context: context,
    barrierDismissible: false,
    builder: (BuildContext dialogContext) {
      return AlertDialog(
        title: const Text('Generation Failed'),
        content: SingleChildScrollView(
          child: Text(errorMessage),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(dialogContext).pop(),
            child: const Text('OK'),
          ),
        ],
      );
    },
  );
}
