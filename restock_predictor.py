import math
import sys

import pandas as pd
from sklearn.linear_model import LinearRegression

RESTOCK_THRESHOLD = 500  # Minimum safe stock level


def predict_restock(csv_file):
    try:
        df = pd.read_csv(csv_file)

        required_columns = {"day", "stock_left"}
        if not required_columns.issubset(df.columns):
            print("❌ CSV must include 'day' and 'stock_left' columns.")
            return
        if df.empty:
            print("❌ CSV contains no inventory records.")
            return

        df = df.copy()
        for column in ("day", "stock_left"):
            df[column] = pd.to_numeric(df[column], errors="coerce")

        values = df[["day", "stock_left"]]
        all_finite = values.apply(lambda column: column.map(math.isfinite)).all().all()
        if values.isna().any().any() or not all_finite:
            print("❌ 'day' and 'stock_left' must contain finite numeric values.")
            return
        if (df["stock_left"] < 0).any():
            print("❌ 'stock_left' cannot contain negative inventory counts.")
            return
        if df["day"].nunique() < 2:
            print("❌ At least two distinct day values are required for a forecast.")
            return

        # Order records explicitly so the forecast always targets the latest day.
        df = df.sort_values("day")
        X = df[["day"]]
        y = df["stock_left"]

        model = LinearRegression()
        model.fit(X, y)

        next_day = [[float(X["day"].max()) + 1]]
        predicted_stock = model.predict(next_day)[0]

        print("\n📦 Inventory Restock Predictor")
        print("----------------------------")
        print(f"Predicted stock for next day: {predicted_stock:.2f}")

        if predicted_stock < RESTOCK_THRESHOLD:
            print("⚠ Warning: Stock will fall below threshold soon. Consider restocking now.")
        else:
            print("✔ Stock levels are safe for now.")

        print("--------------------------------\n")

    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python restock_predictor.py <csv_file>")
        return

    predict_restock(sys.argv[1])


if __name__ == "__main__":
    main()
