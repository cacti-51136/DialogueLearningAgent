"""pytest 公共配置：把 src 加入路径，并提供真实词库 fixture。"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import pytest  # noqa: E402
from dla.config.loader import load_keyword_lib  # noqa: E402


@pytest.fixture(scope="session")
def lib():
    kw_dir = os.path.join(ROOT, "config", "keywords")
    coupling = os.path.join(ROOT, "config", "coupling_rules.yaml")
    return load_keyword_lib(kw_dir, coupling)


@pytest.fixture(scope="session")
def scenario_dir():
    return os.path.join(ROOT, "config", "scenarios")
