// Basic smoke test for the Salary Predictor app.
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:salary_predictor/main.dart';

void main() {
  testWidgets('renders the form with 7 inputs and a Predict button', (WidgetTester tester) async {
    await tester.pumpWidget(const SalaryApp());

    // Two typed fields (work year, remote ratio) plus five dropdowns = seven model inputs.
    expect(find.byType(TextField), findsNWidgets(2));
    expect(find.byWidgetPredicate((w) => w is DropdownButtonFormField), findsNWidgets(5));
    expect(find.text('Predict'), findsOneWidget);
    expect(find.text('Estimate your market salary'), findsOneWidget);
  });
}
