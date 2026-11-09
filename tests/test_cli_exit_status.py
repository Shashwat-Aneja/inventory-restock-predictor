import subprocess
import sys
from pathlib import Path

import pandas as pd

from restock_predictor import main


SCRIPT = Path(__file__).resolve().parents[1] / "restock_predictor.py"


def test_cli_returns_success_for_valid_inventory(tmp_path):
    source = tmp_path / "valid.csv"
    pd.DataFrame({"day": [1, 2], "stock_left": [900, 800]}).to_csv(source, index=False)

    assert main([str(source)]) == 0
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), str(source)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "Stock risk:" in completed.stdout


def test_cli_returns_failure_for_invalid_inventory(tmp_path):
    source = tmp_path / "invalid.csv"
    pd.DataFrame({"day": [1, 2], "stock_left": [900, -5]}).to_csv(source, index=False)

    assert main([str(source)]) == 1
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), str(source)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
    assert "negative inventory counts" in completed.stdout


def test_cli_returns_failure_for_missing_file(tmp_path):
    missing = tmp_path / "missing.csv"
    assert main([str(missing)]) == 1
