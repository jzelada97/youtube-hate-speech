from pathlib import Path

import pytest

DATA = Path("data/raw/youtoxic_english_1000.csv")


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_data: necesita el dataset del cliente en data/raw (no se versiona)")


def pytest_collection_modifyitems(config, items):
    """En CI (o en un clon sin datos) los tests que necesitan el dataset se saltan en lugar de fallar."""
    if DATA.exists():
        return
    skip = pytest.mark.skip(reason=f"falta {DATA} (el dataset no se versiona)")
    for item in items:
        if "requires_data" in item.keywords:
            item.add_marker(skip)
