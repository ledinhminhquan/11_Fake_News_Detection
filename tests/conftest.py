"""Shared pytest fixtures. Tests are CPU-only and never download models/data:
they use the seed corpus + the TF-IDF + LogReg classifier + the lexical stance.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True, scope="session")
def _artifacts_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("fakenews_artifacts")
    os.environ["FAKENEWS_ARTIFACTS_DIR"] = str(d)
    os.environ.setdefault("FAKENEWS_LOG_LEVEL", "WARNING")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    yield


@pytest.fixture
def cfg():
    from fakenews.config import AppConfig
    c = AppConfig()
    c.data.use_hf = False        # offline: seed corpus
    return c
