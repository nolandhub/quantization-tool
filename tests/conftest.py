# ABOUTME: Shared pytest fixtures: path to the real spike fixtures under tests/fixtures.
# ABOUTME: Logs and CSVs captured during the spike are the ground truth for parser and preflight tests.
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures() -> Path:
    return FIXTURES
