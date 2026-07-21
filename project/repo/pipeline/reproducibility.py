"""Central reproducibility and CPU-execution settings."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, clone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "reproducibility.json"
DEFAULT_SEED = 3102
DEFAULT_N_JOBS = 1


@dataclass(frozen=True)
class ReproducibilitySettings:
    """Settings shared by data loading, estimators, and commands."""

    seed: int = DEFAULT_SEED
    n_jobs: int = DEFAULT_N_JOBS

    def __post_init__(self) -> None:
        if not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("seed must be between 0 and 2**32 - 1")
        if self.n_jobs < 1:
            raise ValueError("n_jobs must be at least 1 for CPU execution")


def load_reproducibility_settings(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> ReproducibilitySettings:
    """Load the single saved source of seed and CPU parallelism settings."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return ReproducibilitySettings(
        seed=int(payload["seed"]),
        n_jobs=int(payload["n_jobs"]),
    )


def configure_reproducibility(
    settings: ReproducibilitySettings | None = None,
) -> np.random.Generator:
    """Seed Python and NumPy and constrain common numeric thread pools.

    Call this at a command entry point before loading data or fitting models.
    ``PYTHONHASHSEED`` is also recorded for child processes; a parent process
    must set it before interpreter startup to control that interpreter's hash seed.
    """

    settings = settings or load_reproducibility_settings()
    os.environ["PYTHONHASHSEED"] = str(settings.seed)
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = str(settings.n_jobs)
    random.seed(settings.seed)
    np.random.seed(settings.seed)
    return np.random.default_rng(settings.seed)


def configure_estimator(
    estimator: BaseEstimator,
    settings: ReproducibilitySettings | None = None,
) -> BaseEstimator:
    """Clone an estimator and centrally set all exposed seed/CPU parameters."""

    settings = settings or load_reproducibility_settings()
    configured = clone(estimator)
    updates: dict[str, Any] = {}
    for parameter in configured.get_params(deep=True):
        leaf_name = parameter.rsplit("__", maxsplit=1)[-1]
        if leaf_name == "random_state":
            updates[parameter] = settings.seed
        elif leaf_name == "n_jobs":
            updates[parameter] = settings.n_jobs
    if updates:
        configured.set_params(**updates)
    return configured
