# Verified report assets

Regenerate with:

```powershell
.\.venv\Scripts\python.exe scripts\generate_report_assets.py
```

The generator reads only frozen or previously generated repository artifacts. It does not fit models, rerun held-out evaluation, change labels, or alter prediction files.

- Tables: `report_assets/tables` (CSV source plus IEEE/LaTeX-ready snippets)
- Figures: `report_assets/figures` (PNG, SVG, and source CSV)
- Provenance: `report_assets/asset_manifest.csv`
- Assertions: `report_assets/validation_report.json`

Metric definitions: precision = TP/(TP+FP), recall = TP/(TP+FN), F1 is their harmonic mean, and accuracy = (TP+TN)/N. Target 1 is the positive class. Metrics are unitless in [0,1]; confusion and transition values are tweet counts. Every split and comparison baseline is explicit in the source tables or manifest.
