# Stock Price Prediction Using Multiple Algorithms

A Data Mining course project that predicts the daily closing price of the TATAGLOBAL stock (NSE, India) with three models: Linear Regression, k-Nearest Neighbours regression and an LSTM.
Everything is in `StockPrice.ipynb`.

The code closely follows the Analytics Vidhya tutorial "Stock Prices Prediction Using Machine Learning and Deep Learning Techniques (with Python codes)".

## Data

- Loaded at run time with `quandl.get("NSE/TATAGLOBAL")`: daily Open, High, Low, Last, Close, Total Trade Quantity and Turnover, starting 2010-07-21.
- No data file is included in this repo. **The Quandl NSE dataset has since been retired**, so this call no longer works; you would need to supply a CSV with `Date` and `Close` columns yourself.
- Split by time: the first 987 rows are training data, the rest are validation.

## Models

1. **Linear Regression** (scikit-learn): features are date parts from fastai's `add_datepart` (year, month, week, day, day of week, month/quarter/year start and end flags) plus a `mon_fri` flag for Mondays and Fridays. Target is `Close`.
2. **KNN regression** (scikit-learn `KNeighborsRegressor`): same features, MinMax-scaled, `n_neighbors` 2-9 chosen with `GridSearchCV(cv=5)`.
3. **LSTM** (Keras): input is the previous 60 scaled closing prices, two LSTM layers of 50 units and a Dense(1) output, Adam + MSE, 1 epoch, batch size 1.

## How to run

The notebook was run on Google Colab (Python 3.6).

1. `pip install -r requirements.txt` (needs an old environment because of `fastai==0.7.0`, see below).
2. Replace the `quandl.get(...)` call with your own data source, since the Quandl dataset is retired.
3. Run the cells in order.

## Results (validation RMSE on the closing price, from the committed outputs)

| Model | RMSE |
|---|---|
| Linear Regression | 49.44 |
| KNN | 64.36 |
| LSTM (1 epoch) | 7.83 |

These numbers are not a fair comparison: the LSTM sees the actual previous 60 closing prices for each prediction (one step ahead), while Linear Regression and KNN only see calendar features.

## Known limitations

- The data source (Quandl `NSE/TATAGLOBAL`) is retired, and `fastai.structured.add_datepart` only exists in `fastai==0.7.0`, so the notebook does not run in a current environment as is.
- LSTM: the `MinMaxScaler` is fit on the whole series, including the validation period (data leakage).
- KNN: the validation features are scaled with a separate `fit_transform` instead of the scaler fit on training data, and the grid search uses plain (non-time-ordered) K-fold.
- There is no naive baseline (for example "tomorrow = today"), which a one-step-ahead LSTM should be compared against.
- The LSTM trains for one epoch with no random seed, so results change between runs.
- Execution counts are out of order (for example, the LSTM training log appears under the data-preparation cell), so the committed outputs did not come from a single top-to-bottom run.
- The code uses Keras and has no ensemble.
