import argparse
import math

import pandas as pd
from sklearn.linear_model import LinearRegression

RESTOCK_THRESHOLD = 500  # Minimum safe stock level


def classify_stock_status(predicted_stock, threshold):
    """Classify forecast stock as critical, restock-needed, or healthy."""
    if not math.isfinite(predicted_stock) or predicted_stock < 0:
        raise ValueError("Predicted stock must be a finite, non-negative number.")
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("Restock threshold must be a finite, non-negative number.")
    if predicted_stock < threshold:
        if threshold > 0 and predicted_stock <= threshold * 0.5:
            return "critical"
        return "restock"
    return "healthy"


def estimate_days_until_threshold(current_stock, daily_change, threshold):
    """Estimate days to the threshold from the latest stock and fitted daily trend.

    Returns zero when stock is already at/below the threshold, or None when
    the fitted trend is flat/increasing and therefore does not predict a crossing.
    """
    for label, value in (
        ("Current stock", current_stock),
        ("Daily stock change", daily_change),
        ("Restock threshold", threshold),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{label} must be a finite number.")
    if current_stock < 0 or threshold < 0:
        raise ValueError("Stock and threshold must be non-negative.")
    if current_stock <= threshold:
        return 0
    if daily_change >= 0:
        return None
    return int(math.ceil((current_stock - threshold) / -daily_change))


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
        # Inventory cannot be negative; clamp extrapolated estimates at zero.
        predicted_stock = max(0.0, float(model.predict([[target_day]])[0]))
        daily_change = float(model.coef_[0])
        days_to_threshold = estimate_days_until_threshold(
            float(y.iloc[-1]), daily_change, threshold
        )

        print("\n📦 Inventory Restock Predictor")
        print("----------------------------")
        if days_ahead == 1:
            print(f"Predicted stock for next day: {predicted_stock:.2f}")
        else:
            print(f"Predicted stock for day {target_day:g} ({days_ahead} days ahead): {predicted_stock:.2f}")
        print(f"Estimated daily stock change: {daily_change:+.2f} units/day")

        if days_to_threshold is None:
            print("Estimated days until threshold: no decline indicated by trend")
        else:
            print(f"Estimated days until threshold: {days_to_threshold}")

        status = classify_stock_status(predicted_stock, threshold)
        print(f"Stock risk: {status}")
        if status == "critical":
            print("⚠ Critical: Forecast stock is at or below half the threshold. Prioritize restocking.")
        elif status == "restock":
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
