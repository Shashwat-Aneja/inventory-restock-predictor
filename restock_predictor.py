import pandas as pd
from sklearn.linear_model import LinearRegression
import sys

RESTOCK_THRESHOLD = 500  # Minimum safe stock level

def predict_restock(csv_file):
    try:
        df = pd.read_csv(csv_file)

        if not {"day", "stock_left"}.issubset(df.columns):
            print("❌ CSV must include 'day' and 'stock_left' columns.")
            return

        X = df[["day"]]
        y = df["stock_left"]

        model = LinearRegression()
        model.fit(X, y)

        next_day = [[X.iloc[-1][0] + 1]]
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
