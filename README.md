# Inventory Restock Predictor AI

A machine learning tool that predicts when inventory levels will reach a restocking threshold using historical sales or usage data.  
This project demonstrates predictive analytics to support operational decision-making in business management.

---

## 📊 Features
- Predicts future inventory level using Linear Regression
- Flags restocking risk when the next-day forecast falls below the configured threshold
- Identifies low-inventory risk
- Reads data from CSV file

---

## 📉 Example CSV Format

```
day,stock_left
1,950
2,870
3,800
4,730
5,660
6,600
```

---

## 🚀 Usage

### 1. Clone Repo
```bash
git clone https://github.com/Shashwat-Aneja/inventory-restock-predictor
cd inventory-restock-predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the script
```bash
python restock_predictor.py data.csv

# Override the default minimum safe stock level (500 units)
python restock_predictor.py data.csv --threshold 850
```

### 4. Run the test suite

Install the project dependencies above, then run:

```bash
python -m pytest -q
```

---

## 📁 Project Structure

```
inventory-restock-predictor/
├── restock_predictor.py
├── sample_inventory.csv
├── requirements.txt
├── README.md
└── tests/
    └── test_restock_predictor.py
```

---

## 📌 Future Enhancements
- Predict reorder point date
- Add seasonal trend logic
- Integration with XYLO inventory processing (future scope)

---

Developed by **Shashwat Aneja**


## Forecasting beyond the next day

By default, the predictor forecasts one day after the latest day in the CSV. Use `--days-ahead` to select a later forecast horizon while keeping the same fitted linear-regression model:

```bash
python restock_predictor.py sample_inventory.csv --days-ahead 7
```

The horizon must be a positive whole number. The default of `1` preserves the existing next-day output.


## Forecast bounds

Linear extrapolation can mathematically produce negative values even though physical inventory cannot be negative. The predictor floors the displayed forecast at `0.00`; restock-risk evaluation uses that bounded estimate.
