// Basic smoke test for the Salary Predictor app.
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:salary_predictor/main.dart';

void main() {
  testWidgets('renders the prediction form and Predict button', (WidgetTester tester) async {
    await tester.pumpWidget(const SalaryApp());

    // The single page shows a Predict button and the seven input fields.
    expect(find.text('Predict'), findsOneWidget);
    expect(find.byType(TextField), findsNWidgets(7));
    expect(find.text('Estimate your market salary'), findsOneWidget);
  });
}
