import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;

/// Base URL of the deployed FastAPI service (Task 2).
/// Replace with your Render URL. For an Android emulator hitting a local
/// server use http://10.0.2.2:8000 ; for iOS simulator use http://localhost:8000.
const String kApiBaseUrl = 'https://tech-salary-api.onrender.com';
const String kPredictPath = '/predict';

void main() => runApp(const SalaryApp());

class SalaryApp extends StatelessWidget {
  const SalaryApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Talent Salary Predictor',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF2A7DE1)),
        useMaterial3: true,
      ),
      home: const PredictPage(),
    );
  }
}

class PredictPage extends StatefulWidget {
  const PredictPage({super.key});

  @override
  State<PredictPage> createState() => _PredictPageState();
}

class _PredictPageState extends State<PredictPage> {
  // One controller per model variable (7 inputs = 7 model features).
  final _workYear = TextEditingController(text: '2023');
  final _experience = TextEditingController(text: 'SE');
  final _employment = TextEditingController(text: 'FT');
  final _jobCategory = TextEditingController(text: 'Data Scientist');
  final _companySize = TextEditingController(text: 'M');
  final _companyLocation = TextEditingController(text: 'US');
  final _remoteRatio = TextEditingController(text: '100');

  bool _loading = false;
  String? _resultText;
  bool _isError = false;

  @override
  void dispose() {
    for (final c in [
      _workYear, _experience, _employment, _jobCategory,
      _companySize, _companyLocation, _remoteRatio,
    ]) {
      c.dispose();
    }
    super.dispose();
  }

  Future<void> _predict() async {
    // ---- client-side validation: missing / wrong type ----
    final fields = {
      'work_year': _workYear.text.trim(),
      'experience_level': _experience.text.trim().toUpperCase(),
      'employment_type': _employment.text.trim().toUpperCase(),
      'job_category': _jobCategory.text.trim(),
      'company_size': _companySize.text.trim().toUpperCase(),
      'company_location_grp': _companyLocation.text.trim().toUpperCase(),
      'remote_ratio': _remoteRatio.text.trim(),
    };
    final missing = fields.entries.where((e) => e.value.isEmpty).map((e) => e.key).toList();
    if (missing.isNotEmpty) {
      _show('Missing value(s): ${missing.join(', ')}', isError: true);
      return;
    }
    final year = int.tryParse(fields['work_year']!);
    final remote = int.tryParse(fields['remote_ratio']!);
    if (year == null || remote == null) {
      _show('work_year and remote_ratio must be whole numbers.', isError: true);
      return;
    }

    final body = <String, dynamic>{
      'work_year': year,
      'experience_level': fields['experience_level'],
      'employment_type': fields['employment_type'],
      'job_category': fields['job_category'],
      'company_size': fields['company_size'],
      'company_location_grp': fields['company_location_grp'],
      'remote_ratio': remote,
    };

    setState(() => _loading = true);
    try {
      final resp = await http
          .post(
            Uri.parse('$kApiBaseUrl$kPredictPath'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode(body),
          )
          .timeout(const Duration(seconds: 30));

      final decoded = jsonDecode(resp.body);
      if (resp.statusCode == 200) {
        final value = (decoded['predicted_salary_usd'] as num).toDouble();
        _show('Predicted annual salary:\n\$${_fmt(value)} USD\n'
            '(model: ${decoded['model_used']})');
      } else {
        _show('The values are out of range or invalid.\n\n${_readableError(decoded)}',
            isError: true);
      }
    } catch (e) {
      _show('Could not reach the API. Check the URL / connection.\n$e', isError: true);
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  String _readableError(dynamic decoded) {
    try {
      final detail = decoded['detail'];
      if (detail is List) {
        return detail
            .map((d) => '• ${(d['loc'] as List).last}: ${d['msg']}')
            .join('\n');
      }
      return detail.toString();
    } catch (_) {
      return decoded.toString();
    }
  }

  String _fmt(double v) {
    final s = v.round().toString();
    final buf = StringBuffer();
    for (int i = 0; i < s.length; i++) {
      if (i > 0 && (s.length - i) % 3 == 0) buf.write(',');
      buf.write(s[i]);
    }
    return buf.toString();
  }

  void _show(String text, {bool isError = false}) {
    setState(() {
      _resultText = text;
      _isError = isError;
    });
  }

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Scaffold(
      appBar: AppBar(
        title: const Text('Talent Salary Predictor'),
        backgroundColor: scheme.primary,
        foregroundColor: scheme.onPrimary,
      ),
      body: SafeArea(
        child: Center(
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 560),
            child: SingleChildScrollView(
              padding: const EdgeInsets.all(20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  Text('Estimate your market salary',
                      style: Theme.of(context).textTheme.headlineSmall
                          ?.copyWith(fontWeight: FontWeight.bold)),
                  const SizedBox(height: 4),
                  Text('Enter a professional profile to predict the expected annual salary (USD).',
                      style: Theme.of(context).textTheme.bodyMedium
                          ?.copyWith(color: Colors.grey.shade600)),
                  const SizedBox(height: 20),
                  Card(
                    elevation: 2,
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                    child: Padding(
                      padding: const EdgeInsets.all(16),
                      child: Column(
                        children: [
                          _numField(_workYear, 'Work year', 'Year 2020-2027, e.g. 2023'),
                          _textField(_experience, 'Experience level', 'EN, MI, SE or EX'),
                          _textField(_employment, 'Employment type', 'FT, PT, CT or FL'),
                          _textField(_jobCategory, 'Job category',
                              'Data Engineer / Data Scientist / Data Analyst / ML/AI Engineer / Management / Other'),
                          _textField(_companySize, 'Company size', 'S, M or L'),
                          _textField(_companyLocation, 'Company location',
                              'US, GB, CA, ES, IN, DE or Other'),
                          _numField(_remoteRatio, 'Remote ratio', '0, 50 or 100'),
                        ],
                      ),
                    ),
                  ),
                  const SizedBox(height: 16),
                  SizedBox(
                    height: 52,
                    child: FilledButton.icon(
                      onPressed: _loading ? null : _predict,
                      icon: _loading
                          ? const SizedBox(
                              width: 20, height: 20,
                              child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white))
                          : const Icon(Icons.calculate_outlined),
                      label: Text(_loading ? 'Predicting...' : 'Predict',
                          style: const TextStyle(fontSize: 16)),
                    ),
                  ),
                  const SizedBox(height: 20),
                  if (_resultText != null) _resultCard(scheme),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _resultCard(ColorScheme scheme) {
    final bg = _isError ? Colors.red.shade50 : Colors.green.shade50;
    final fg = _isError ? Colors.red.shade900 : Colors.green.shade900;
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: bg,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: _isError ? Colors.red.shade200 : Colors.green.shade200),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(_isError ? Icons.error_outline : Icons.check_circle_outline, color: fg),
          const SizedBox(width: 12),
          Expanded(
            child: Text(_resultText!,
                style: TextStyle(color: fg, fontSize: 16, fontWeight: FontWeight.w600)),
          ),
        ],
      ),
    );
  }

  Widget _numField(TextEditingController c, String label, String helper) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 8),
        child: TextField(
          controller: c,
          keyboardType: TextInputType.number,
          inputFormatters: [FilteringTextInputFormatter.digitsOnly],
          decoration: _decoration(label, helper),
        ),
      );

  Widget _textField(TextEditingController c, String label, String helper) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 8),
        child: TextField(
          controller: c,
          textCapitalization: TextCapitalization.characters,
          decoration: _decoration(label, helper),
        ),
      );

  InputDecoration _decoration(String label, String helper) => InputDecoration(
        labelText: label,
        helperText: helper,
        helperMaxLines: 2,
        border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
        isDense: true,
      );
}
