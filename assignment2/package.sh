#!/usr/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

ZIP_NAME="${1:-submission_assignment2.zip}"

# Manually selected files required by ASSIGNMENT.md
MANUAL_FILES=(
  "README.md"
  "REPORT.md"
  "chat.md"
  "pyproject.toml"
  "uv.lock"
  "run_all.py"
  "src/plant_pathology/__init__.py"
  "src/plant_pathology/data.py"
  "src/plant_pathology/models.py"
  "src/plant_pathology/metrics.py"
  "src/plant_pathology/train.py"
  "src/plant_pathology/evaluate.py"
  "src/plant_pathology/predict.py"
  "src/plant_pathology/major_voting.py"
  "src/plant_pathology/validate_voting.py"
  "src/plant_pathology/validate_submission.py"
  "tests/test_public.py"
)

# Required outputs from ASSIGNMENT.md
REQUIRED_OUTPUTS=(
  "results/experiments.csv"
  "results/error_analysis.csv"
  "results/training_curves_baseline.png"
  "results/confusion_matrix_baseline.png"
  "results/confusion_matrix_final.png"
  "predictions/submission.csv"
)

mapfile -t CONFIG_FILES < <(find configs -maxdepth 1 -type f -name '*.json' | sort)

if [[ "${#CONFIG_FILES[@]}" -eq 0 ]]; then
  echo "No config files found under configs/." >&2
  exit 1
fi

missing=0
for path in "${MANUAL_FILES[@]}" "${CONFIG_FILES[@]}" "${REQUIRED_OUTPUTS[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    missing=1
  fi
done

if [[ "$missing" -ne 0 ]]; then
  echo "Packaging aborted due to missing required files." >&2
  exit 1
fi

tmp_list="$(mktemp)"
trap 'rm -f "$tmp_list"' EXIT

# 1) Required manual files + required outputs
printf '%s\n' "${MANUAL_FILES[@]}" "${CONFIG_FILES[@]}" "${REQUIRED_OUTPUTS[@]}" > "$tmp_list"

# 2) Add non-hack files from results/ and predictions/ (exclude .gitkeep and zip files)
find results predictions -maxdepth 1 -type f \
  ! -name '.gitkeep' \
  ! -iname '*hack*' \
  ! -name '*.zip' \
  -print >> "$tmp_list"

# De-duplicate and keep stable order
awk '!seen[$0]++' "$tmp_list" > "${tmp_list}.dedup"
mv "${tmp_list}.dedup" "$tmp_list"

rm -f "$ZIP_NAME"
zip -r "$ZIP_NAME" -@ < "$tmp_list"

echo "Created: $ZIP_NAME"
echo "Included files:"
wc -l < "$tmp_list"

# Safety check: packaged contents should not include hack-related files
if zipinfo -1 "$ZIP_NAME" | grep -Eiq 'hack'; then
  echo "Warning: archive contains paths matching 'hack'. Please inspect:"
  zipinfo -1 "$ZIP_NAME" | grep -Ei 'hack'
  exit 1
fi

echo "Sanity check passed: no hack-related paths in archive."
