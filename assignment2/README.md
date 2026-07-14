# Assignment 2 Starter

Read `ASSIGNMENT.md` before starting. Use Python 3.11 or 3.12 in an isolated environment.

## Setup

With `uv`:

```bash
uv sync --extra dev
```

Or with `venv` and `pip`:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
```

Extract `assignment2-course-data.zip` here. You should then have:

```text
data/train.csv
data/validation.csv
data/test.csv
data/images/
```

## Check the Starter

```bash
python -m pytest
```

## Commands to Support

After completing the `TODO` sections, your code should support:

```bash
python -m plant_pathology.train --config configs/baseline.json
python -m plant_pathology.evaluate --config configs/baseline.json --checkpoint checkpoints/best.pt
python -m plant_pathology.predict --config configs/baseline.json --checkpoint checkpoints/best.pt
python -m plant_pathology.validate_submission
```

The starter functions marked `TODO` are intentionally incomplete. You may change the code and
configuration files, but keep the required outputs described in `ASSIGNMENT.md`.
