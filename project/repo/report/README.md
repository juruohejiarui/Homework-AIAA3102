# IEEE conference report source

`main.tex` uses the standard IEEE conference class:

```tex
\documentclass[conference]{IEEEtran}
```

The source imports verified LaTeX tables and PNG figures from `../report_assets/`. The pinned build script downloads the official Tectonic Windows release, verifies its SHA-256 digest, compiles the bibliography and document, and places the result at `../report.pdf`:

```powershell
powershell -ExecutionPolicy Bypass -File ..\scripts\build_report.ps1
```

The verified author names are LAI Jiaxing and HE Jiarui. Affiliation, location, and email were not supplied and are intentionally omitted rather than guessed.

Quantitative provenance is recorded in:

- `../report_assets/asset_manifest.csv`
- `../report_assets/validation_report.json`
- `../results/report_evidence_matrix.csv`

The report must continue to use the machine-generated Ticket 4 ID 767 scores (`0.4876230020837692` to `0.5448153095708489`) and must not use the stale scores in the ticket narrative.
