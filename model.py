import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv("Grain_Prices_20250211.csv")

df = pd.read_csv("Grain_Prices_20250211.csv", parse_dates=['Date'], index_col='Date')
df = df.sort_index()  # Make sure the data is sorted by Date

bid=df['Bid']

print(df.info())
print(df.describe())

result = adfuller(bid)
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')

plot_acf(bid, lags=50)
plt.show()
plot_pacf(bid, lags=50)
plt.show()

df['Bid_diff'] = bid.diff().dropna()

result = adfuller(df['Bid_diff'].dropna())
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')


model = SARIMAX(bid, 
                order=(1, 1, 1),  # p, d, q
                seasonal_order=(1, 1, 1, 12),  # P, D, Q, s
                enforce_stationarity=False, 
                enforce_invertibility=False)

results = model.fit()

print(results.summary())

results.plot_diagnostics(figsize=(10, 8))
plt.show()

forecast = results.get_forecast(steps=12)
forecast_index = pd.date_range(start=df.index[-1], periods=13, freq='M')[1:]