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

Inventory regression uses day offsets relative to the first observation, avoiding large-intercept cancellation when valid day indexes are numerically large.

An estimated threshold crossing is at least one day when current stock is still above the threshold, even when floating-point tolerance places the positive crossing very close to zero.

`--json` prints only the structured forecast summary on success, including target day, stock risk and threshold-crossing estimate. Validation failures retain a nonzero exit status.

`--delimiter ";"`, `--delimiter tab` and `--delimiter "|"` support alternate input CSV separators, including quoted fields. Duplicate-header validation uses the selected separator. The default input and all CSV exports remain comma-separated.

`--lookback-points N` fits the most recent N distinct observed days after duplicate-day aggregation. The summary reports selected and available history counts. This is an observation window, not necessarily N consecutive calendar days; the default uses all history.

Seekable caller-owned text and UTF-8 binary CSV streams now receive the same duplicate-header checks as file paths, before pandas can rename ambiguous columns. Validation restores the starting position and leaves caller streams open; nonseekable stream objects are rejected explicitly.

Structured results and human reports include `historical_fit_mae` in stock units for the selected history window. It is explicitly an in-sample fit diagnostic, not an out-of-sample accuracy estimate or a guarantee of future inventory levels. Computation stays scaled to support large finite stock values.

CSV paths ending in `.gz`, `.bz2`, or `.xz` are supported without unpacking a separate file. Header validation reads the same compression format as pandas and still rejects duplicate columns, including BOM-prefixed quoted headers and alternate delimiters. Archive containers such as ZIP/TAR are outside this feature.

`--output forecast.json` atomically persists the complete structured inventory result. The output must differ from the source (including hard-link aliases). A failed write/replace preserves the previous report and cleans temporary files.

`--lead-time-days N` adds `reorder_before_delivery`, comparing projected days until the configured threshold with a non-negative supplier lead time. It is a trend-based planning flag; no order or hardware operation is performed. No declining crossing yields false; already-low stock has zero runway.

Corrupt compressed bodies, invalid DEFLATE blocks and header-parser limit failures use the normal validation-error path rather than escaping as EOFError, zlib.error or csv.Error. Header inspection restores caller stream position/ownership on these failures. No invalid input is exported.

### Optional raw CSV row budget

Use `--max-input-rows 100000` to reject an input file with more than that many data rows before modeling or export. The budget is per file, including comparison/training/scoring inputs, and applies before filtering or aggregation. The default remains unlimited. Values must be integers from 1 to 10,000,000. Only the budget plus one rows are loaded; oversized inputs fail rather than being silently truncated. This is a row limit, not a byte or field-size limit. Python callers can use `max_rows=`.

`--timeline-days 14` includes daily stock forecasts and risk statuses from one fitted trend (1..3660 days). Negative predictions clamp to zero. The main `--days-ahead` point forecast remains independent; timeline output is in the returned/JSON summary.

`--duplicate-days mean|last|error` controls repeated-day inventory observations. Default mean preserves equal day weighting; last uses the final source row per day before sorting; error rejects duplicates before fitting. Nondefault policy is recorded in JSON.

`--encoding utf-8|utf-8-sig|cp1252|latin-1` explicitly selects CSV input decoding, including compressed files and binary streams. Header validation and data parsing use the same encoding; comparison/training/scoring inputs share it. The default remains UTF-8, and export encoding is unchanged. Text streams are already decoded by their caller.

`--max-input-bytes N` snapshots each decompressed CSV input into a bounded byte buffer before parsing (1..64 MiB). Oversized input fails; gzip/bzip2/xz limits apply after decompression. Caller-owned seekable streams retain position and ownership. Text stream bytes use the selected encoding. This optional mode requires local regular files or seekable streams and composes with the raw row budget.

`--expect-input-sha256 HEX` requires the main CSV input to match a SHA-256 digest of its decompressed bytes, including any BOM, before header/data parsing. The verified byte snapshot is what gets parsed. It uses the explicit byte budget or a 64 MiB cap by default. For text streams, bytes use the selected encoding. This flag applies to the main/training file; comparison and separate scoring files are not implicitly required to have the same digest. Python readers accept `expected_sha256=` independently per file.
