# Inventory Restock Predictor AI

A machine learning tool that predicts when inventory levels will reach a restocking threshold using historical sales or usage data.  
This project demonstrates predictive analytics to support operational decision-making in business management.

---

## 📊 Features
- Predicts future inventory level using Linear Regression
- Estimates when to reorder stock
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
pip install pandas scikit-learn
```

### 3. Run the script
```bash
python restock_predictor.py data.csv
```

---

## 📁 Project Structure

```
inventory-restock-predictor/
│
├── restock_predictor.py
└── README.md
```

---

## 📌 Future Enhancements
- Predict reorder point date
- Add seasonal trend logic
- Integration with XYLO inventory processing (future scope)

---

Developed by **Shashwat Aneja**
