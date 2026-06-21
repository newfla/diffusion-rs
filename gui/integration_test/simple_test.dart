import 'package:flutter_test/flutter_test.dart';
import 'package:diffusion_rs_gui/app.dart';
import 'package:diffusion_rs_gui/src/rust/frb_generated.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:integration_test/integration_test.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();
  setUpAll(() async => await RustLib.init());
  testWidgets('App launches and shows Generate button',
      (WidgetTester tester) async {
    await tester.pumpWidget(
      const ProviderScope(child: DiffusionRsApp()),
    );
    await tester.pumpAndSettle();
    expect(find.text('Generate'), findsOneWidget);
  });
}
