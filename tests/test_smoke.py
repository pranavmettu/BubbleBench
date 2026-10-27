"""Smoke test: the package installs and imports correctly."""

from importlib.metadata import version

import bubblebench


def test_version_matches_installed_metadata() -> None:
    """The version in __init__.py matches the one in pyproject.toml."""
    assert bubblebench.__version__ == version("bubblebench")
