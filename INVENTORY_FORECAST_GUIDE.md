Inventory forecast guide

The CSV must contain finite numeric `day` and `stock_left` values. Negative stock is rejected and at least two distinct days are required. Repeated observations for one day are averaged before regression.

Run `python restock_predictor.py sample_inventory.csv --threshold 500 --days-ahead 2`.

This is a straight-line extrapolation; it does not model incoming deliveries, lead times, seasonal demand, or intermittent demand. Treat the threshold as a configurable alert level, not an optimized reorder point. Validate forecast errors and incorporate supplier lead time before relying on the output for purchasing.