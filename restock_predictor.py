import argparse
import math

import pandas as pd
from sklearn.linear_model import LinearRegression

RESTOCK_THRESHOLD = 500  # Minimum safe stock level


def predict_restock(csv_file, threshold=RESTOCK_THRESHOLD, days_ahead=1):
    if not math.isfinite(threshold) or threshold < 0:
        print("❌ Restock threshold must be a finite, non-negative number.")
        return
    if not isinstance(days_ahead, int) or isinstance(days_ahead, bool) or days_ahead < 1:
        print("❌ Forecast horizon must be a positive whole number of days.")
        return

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

        # Aggregate repeated observations for a day so duplicate rows do not
        # give that day extra weight in the regression model.
        df = (
            df.groupby("day", as_index=False)["stock_left"]
            .mean()
            .sort_values("day")
        )
        X = df[["day"]]
        y = df["stock_left"]

        model = LinearRegression()
        model.fit(X, y)

        target_day = float(X["day"].max()) + days_ahead
        predicted_stock = model.predict([[target_day]])[0]

        print("\n📦 Inventory Restock Predictor")
        print("----------------------------")
        if days_ahead == 1:
            print(f"Predicted stock for next day: {predicted_stock:.2f}")
        else:
            print(f"Predicted stock for day {target_day:g} ({days_ahead} days ahead): {predicted_stock:.2f}")

        if predicted_stock < threshold:
            print("⚠ Warning: Stock will fall below threshold soon. Consider restocking now.")
        else:
            print("✔ Stock levels are safe for now.")

        print("--------------------------------\n")

    except (OSError, pd.errors.ParserError, ValueError) as e:
        print(f"❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Forecast next-day inventory and flag low-stock risk."
    )
    parser.add_argument("csv_file", help="Path to inventory CSV data")
    parser.add_argument(
        "--threshold",
        type=float,
        default=RESTOCK_THRESHOLD,
        help=f"Minimum safe stock level (default: {RESTOCK_THRESHOLD})",
    )
    parser.add_argument(
        "--days-ahead",
        type=int,
        default=1,
        help="Number of days beyond the latest observation to forecast (default: 1)",
    )
    args = parser.parse_args()
    if not math.isfinite(args.threshold) or args.threshold < 0:
        parser.error("--threshold must be a finite, non-negative number")
    if args.days_ahead < 1:
        parser.error("--days-ahead must be a positive whole number")
    predict_restock(args.csv_file, threshold=args.threshold, days_ahead=args.days_ahead)


if __name__ == "__main__":
    main()
