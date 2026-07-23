// Basic smoke test for the Salary Predictor app.
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:salary_predictor/main.dart';

void main() {
  testWidgets('renders seven text fields and a Predict button', (WidgetTester tester) async {
    await tester.pumpWidget(const SalaryApp());

    // One text field per model variable.
    expect(find.byType(TextField), findsNWidgets(7));
    expect(find.text('Predict'), findsOneWidget);
    expect(find.text('Estimate your market salary'), findsOneWidget);
  });
}
