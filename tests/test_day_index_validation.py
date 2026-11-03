import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import pandas as pd

from restock_predictor import predict_restock


class InventoryDayIndexTests(unittest.TestCase):
    def test_fractional_day_indexes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "inventory.csv"
            pd.DataFrame(
                {"day": [1, 2.5, 3], "stock_left": [100, 90, 80]}
            ).to_csv(csv_path, index=False)

            output = StringIO()
            with redirect_stdout(output):
                result = predict_restock(csv_path)

            self.assertIsNone(result)
            self.assertIn("whole-number day indexes", output.getvalue())


if __name__ == "__main__":
    unittest.main()
