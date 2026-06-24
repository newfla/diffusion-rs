import '../../../shared/models/progress_event.dart';
import 'generation_service.dart';

/// Mock implementation of [GenerationService] for Phase 1 (per MOCK-01).
///
/// Uses an async* generator (NOT Timer.periodic per CONTEXT.md anti-patterns)
/// to emit 20 progress events with ~250ms delay each, totaling ~5 seconds
/// (per MOCK-02). The stream naturally completes and cleans up when cancelled.
class MockGenerationService implements GenerationService {
  @override
  Stream<ProgressEvent> generate(Map<String, dynamic> params) async* {
    const totalSteps = 20;
    for (var i = 1; i <= totalSteps; i++) {
      await Future.delayed(const Duration(milliseconds: 250));
      yield ProgressEvent(
        step: i,
        steps: totalSteps,
        time: i * 0.25,
      );
    }
  }
}
