import gzip
import bz2
import lzma
import tempfile
import io
import csv
import os
from pathlib import Path
import argparse
import json
import math
from numbers import Real

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

RESTOCK_THRESHOLD = 500  # Minimum safe stock level
MAX_SAFE_DAY = 2**53 - 1  # Largest integer exactly represented by float64


def _read_csv_input(csv_file, delimiter=",", **options):
    if delimiter not in (",", ";", "\t", "|"):
        raise ValueError("Unsupported input delimiter.")
    _reject_duplicate_headers(csv_file, delimiter)
    try:
        frame = pd.read_csv(csv_file, **options, **({"sep": delimiter} if delimiter != "," else {}))
    except (EOFError, lzma.LZMAError) as exc:
        raise ValueError("Compressed CSV input is corrupt or incomplete.") from exc
    return frame


def _reject_duplicate_headers(csv_file, delimiter=","):
    """Inspect original UTF-8 CSV headers without taking ownership of streams."""
    def header_lines(source):
        # csv.reader needs physical lines even for StringIO objects whose default
        # iterator does not recognize CR-only records. Preserve quoted newlines.
        pending = ""
        first = True
        started = False
        while True:
            chars = []
            while True:
                char = pending or source.read(1)
                pending = ""
                if not char:
                    break
                if first:
                    first = False
                    if char == "\ufeff":
                        continue
                chars.append(char)
                if char == "\r":
                    following = source.read(1)
                    if following == "\n":
                        chars.append(following)
                    else:
                        pending = following
                    break
                if char == "\n":
                    break
            if not chars:
                return
            line = "".join(chars)
            if not started and not line.strip():
                continue
            started = True
            yield line

    if isinstance(csv_file, (str, os.PathLike)):
        if not Path(csv_file).is_file():
            return
        opener = {".gz": gzip.open, ".bz2": bz2.open, ".xz": lzma.open}.get(Path(csv_file).suffix, open)
        try:
            with opener(csv_file, mode="rt", encoding="utf-8-sig", newline="") as source:
                headers = next(csv.reader(header_lines(source), delimiter=delimiter), [])
        except (EOFError, lzma.LZMAError) as exc:
            raise ValueError("Compressed CSV input is corrupt or incomplete.") from exc
    else:
        if not all(hasattr(csv_file, name) for name in ("read", "seek", "tell")) or (hasattr(csv_file, "seekable") and not csv_file.seekable()):
            raise ValueError("CSV streams must be seekable for header validation.")
        position = csv_file.tell()
        binary = isinstance(csv_file.read(0), bytes)
        source = io.TextIOWrapper(csv_file, encoding="utf-8-sig", newline="") if binary else csv_file
        try:
            headers = next(csv.reader(header_lines(source), delimiter=delimiter), [])
        finally:
            if binary:
                source.detach()
            csv_file.seek(position)
    if len(headers) != len(set(headers)):
        raise ValueError("CSV column names must be unique; duplicate headers are ambiguous.")


def _finite_number(value, label):
    """Validate a real numeric input without accepting booleans or numeric strings."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be a finite number.")
    return numeric


def classify_stock_status(predicted_stock, threshold):
    """Classify forecast stock as critical, restock-needed, or healthy."""
    predicted_stock = _finite_number(predicted_stock, "Predicted stock")
    threshold = _finite_number(threshold, "Restock threshold")
    if predicted_stock < 0:
        raise ValueError("Predicted stock must be a finite, non-negative number.")
    if threshold < 0:
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
    current_stock = _finite_number(current_stock, "Current stock")
    daily_change = _finite_number(daily_change, "Daily stock change")
    threshold = _finite_number(threshold, "Restock threshold")
    if current_stock < 0 or threshold < 0:
        raise ValueError("Stock and threshold must be non-negative.")
    if current_stock <= threshold:
        return 0
    if daily_change >= 0:
        return None

    raw_days = (current_stock - threshold) / -daily_change
    if not math.isfinite(raw_days):
        raise ValueError("Threshold crossing exceeds the supported numeric range.")
    nearest_integer = round(raw_days)
    # Regression coefficients can differ from an exact integer by a few ULPs.
    # Avoid turning an exact three-day crossing into four days because of noise.
    if math.isclose(raw_days, nearest_integer, rel_tol=0.0, abs_tol=1e-12):
        return max(1, int(nearest_integer))
    return int(math.ceil(raw_days))


def predict_restock(csv_file, threshold=RESTOCK_THRESHOLD, days_ahead=1, json_output=False, delimiter=",", lookback_points=None):
    """Print the forecast and return a structured summary for downstream use."""
    try:
        threshold = _finite_number(threshold, "Restock threshold")
    except ValueError:
        print("❌ Restock threshold must be a finite, non-negative number.")
        return None
    if threshold < 0:
        print("❌ Restock threshold must be a finite, non-negative number.")
        return None
    if not isinstance(days_ahead, int) or isinstance(days_ahead, bool) or days_ahead < 1:
        print("❌ Forecast horizon must be a positive whole number of days.")
        return None

    if lookback_points is not None and (not isinstance(lookback_points, int) or isinstance(lookback_points, bool) or lookback_points < 2):
        print("❌ lookback_points must be an integer of at least two.")
        return None
    try:
        df = _read_csv_input(csv_file, delimiter)

        required_columns = {"day", "stock_left"}
        if not required_columns.issubset(df.columns):
            print("❌ CSV must include 'day' and 'stock_left' columns.")
            return None
        if df.empty:
            print("❌ CSV contains no inventory records.")
            return None

        if df[["day", "stock_left"]].apply(lambda column: column.map(lambda value: isinstance(value, (bool, np.bool_)))).any().any():
            raise ValueError("Inventory values must be numeric, not booleans.")
        df = df.copy()
        for column in ("day", "stock_left"):
            df[column] = pd.to_numeric(df[column], errors="coerce")

        values = df[["day", "stock_left"]]
        all_finite = values.apply(lambda column: column.map(math.isfinite)).all().all()
        if values.isna().any().any() or not all_finite:
            print("❌ 'day' and 'stock_left' must contain finite numeric values.")
            return None
        if any(not float(day).is_integer() for day in df["day"]):
            print("❌ 'day' must contain whole-number day indexes.")
            return None
        if df["day"].map(lambda value: abs(int(value)) > MAX_SAFE_DAY).any():
            print("❌ Day indexes must be within the exact float64 integer range.")
            return None
        if (df["stock_left"] < 0).any():
            print("❌ 'stock_left' cannot contain negative inventory counts.")
            return None
        if df["day"].nunique() < 2:
            print("❌ At least two distinct day values are required for a forecast.")
            return None

        # Aggregate repeated observations for a day so duplicate rows do not
        # give that day extra weight in the regression model.
        df = (
            df.groupby("day", as_index=False)["stock_left"]
            .agg(lambda values: float(values.max()) * (math.fsum(float(value / values.max()) for value in values) / len(values)) if values.max() else 0.0)
            .sort_values("day")
        )
        available_history_points = len(df)
        if lookback_points is not None:
            df = df.tail(lookback_points).copy()
        X = df[["day"]]
        y = df["stock_left"]

        model = LinearRegression()
        origin_day = float(X["day"].iloc[0])
        stock_scale = float(y.max()) or 1.0
        model.fit(X - origin_day, y / stock_scale)

        latest_day = int(X["day"].max())
        if latest_day + days_ahead > MAX_SAFE_DAY or latest_day + days_ahead < -MAX_SAFE_DAY:
            print("❌ Forecast target day exceeds the exact float64 integer range.")
            return None
        target_day = latest_day + days_ahead
        # Inventory cannot be negative; clamp extrapolated estimates at zero.
        raw_prediction = float(model.predict(pd.DataFrame({"day": [target_day - origin_day]}))[0]) * stock_scale
        if not math.isfinite(raw_prediction):
            raise ValueError("Forecast stock must be a finite number.")
        fitted_normalized = model.predict(X - origin_day)
        normalized_residuals = y.to_numpy(dtype=float) / stock_scale - fitted_normalized
        historical_fit_mae = stock_scale * (math.fsum(abs(float(value)) for value in normalized_residuals) / len(y))
        if not math.isfinite(historical_fit_mae):
            raise ValueError("Historical fit diagnostics exceed the supported range.")
        predicted_stock = max(0.0, raw_prediction)
        daily_change = float(model.coef_[0]) * stock_scale
        days_to_threshold = estimate_days_until_threshold(
            float(y.iloc[-1]), daily_change, threshold
        )
        status = classify_stock_status(predicted_stock, threshold)
        summary = {
            "target_day": float(target_day),
            "predicted_stock": float(predicted_stock),
            "daily_stock_change": float(daily_change),
            "days_until_threshold": days_to_threshold,
            "restock_threshold": float(threshold),
            "status": status,
            "history_points": int(len(df)),
            "historical_fit_mae": float(historical_fit_mae),
            "available_history_points": int(available_history_points),
        }

        if json_output:
            print(json.dumps(summary, allow_nan=False))
            return summary

        print("\n📦 Inventory Restock Predictor")
        print("----------------------------")
        if days_ahead == 1:
            print(f"Predicted stock for next day: {predicted_stock:.2f}")
        else:
            print(f"Predicted stock for day {target_day:g} ({days_ahead} days ahead): {predicted_stock:.2f}")
        print(f"Estimated daily stock change: {daily_change:+.2f} units/day")
        print(f"Historical fit MAE (in-sample): {historical_fit_mae:.2f} units")

        if days_to_threshold is None:
            print("Estimated days until threshold: no decline indicated by trend")
        else:
            print(f"Estimated days until threshold: {days_to_threshold}")

        print(f"Stock risk: {status}")
        if status == "critical":
            print("⚠ Critical: Forecast stock is at or below half the threshold. Prioritize restocking.")
        elif status == "restock":
            print("⚠ Warning: Stock will fall below threshold soon. Consider restocking now.")
        else:
            print("✔ Stock levels are safe for now.")

        print("--------------------------------\n")
        return summary

    except (OSError, pd.errors.ParserError, ValueError) as e:
        print(f"❌ Error: {e}")
        return None


def export_summary(summary, destination, source):
    destination = Path(destination).expanduser().resolve()
    source = Path(source).expanduser().resolve()
    if destination == source or (destination.exists() and destination.samefile(source)):
        raise ValueError("JSON output must differ from the inventory input.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(summary, handle, allow_nan=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
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
    parser.add_argument("--json", action="store_true", help="Print only the structured forecast summary as JSON on success")
    parser.add_argument("--delimiter", choices=(",", ";", "tab", "|"), default=",", help="Input CSV delimiter; exports remain comma-separated")
    parser.add_argument("--lookback-points", type=int, help="Fit only the latest N distinct observed days, after daily aggregation")
    parser.add_argument("--output", help="Atomically save the structured forecast as JSON, distinct from the input")
    args = parser.parse_args(argv)
    delimiter = "\t" if args.delimiter == "tab" else args.delimiter
    if not math.isfinite(args.threshold) or args.threshold < 0:
        parser.error("--threshold must be a finite, non-negative number")
    if args.days_ahead < 1:
        parser.error("--days-ahead must be a positive whole number")
    options = {"json_output": True} if args.json else {}
    if args.lookback_points is not None:
        options["lookback_points"] = args.lookback_points
    if delimiter != ",":
        options["delimiter"] = delimiter
    result = predict_restock(args.csv_file, threshold=args.threshold, days_ahead=args.days_ahead, **options)
    if result is None:
        return 1
    if args.output:
        try:
            export_summary(result, args.output, args.csv_file)
        except (OSError, ValueError) as exc:
            print(f"Error exporting inventory summary: {exc}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
