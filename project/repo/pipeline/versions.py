"""Capture runtime and package versions for reproducible experiment artifacts."""

from __future__ import annotations

import platform
from importlib import metadata
from pathlib import Path
from typing import Sequence

from .artifacts import write_json_artifact

DEFAULT_PACKAGES = ("numpy", "pandas", "scikit-learn", "matplotlib")


def capture_package_versions(
    packages: Sequence[str] = DEFAULT_PACKAGES,
) -> dict[str, object]:
    package_versions: dict[str, str | None] = {}
    for package in packages:
        try:
            package_versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            package_versions[package] = None
    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "platform": platform.platform(),
        "packages": package_versions,
    }


def write_package_versions(
    path: str | Path,
    packages: Sequence[str] = DEFAULT_PACKAGES,
) -> Path:
    return write_json_artifact(capture_package_versions(packages), path)
