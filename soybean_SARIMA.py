import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import adfuller

df = pd.read_csv("Soybeans_Monthly.csv")

df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'], format='%Y-%B')

df = df.set_index('Date')

df = df.sort_index()

print(df.info())
print(df.describe())

result = adfuller(df['Price'])
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')

plot_acf(df['Price'], lags=50)
plt.show()
plot_pacf(df['Price'], lags=50)
plt.show()

df['Price_diff'] = df['Price'].diff().dropna()
result = adfuller(df['Price_diff'].dropna())
print(f'ADF Statistic (Differenced): {result[0]}')
print(f'p-value (Differenced): {result[1]}')

train = df[:-12]
test = df[-12:]

model = SARIMAX(train['Price'],
                order=(1, 1, 1),  # p, d, q
                seasonal_order=(1, 1, 1, 12),  # P, D, Q, s
                enforce_stationarity=False,
                enforce_invertibility=False)

results = model.fit()

print(results.summary())

results.plot_diagnostics(figsize=(10, 8))
plt.show()

forecast = results.get_forecast(steps=12)
forecast_index = pd.date_range(start=test.index[0], periods=13, freq='M')[1:]

forecast_mean = forecast.predicted_mean
forecast_conf_int = forecast.conf_int()

plt.figure(figsize=(12, 6))
plt.plot(df['Price'], label='Historical Data', linewidth=3, c='purple')
plt.plot(forecast_index, forecast_mean, label='Forecasted Data', linestyle='--', color='green', linewidth=3)
plt.fill_between(forecast_index, forecast_conf_int.iloc[:, 0], forecast_conf_int.iloc[:, 1], color='green', alpha=0.2)
plt.title("Soybean Bids Forecast")
plt.xlabel("Date")
plt.ylabel("Bid")
plt.legend()
plt.show()

actuals = test['Price']  
forecasted = forecast_mean

mae = mean_absolute_error(actuals, forecasted)
mse = mean_squared_error(actuals, forecasted)
print(f'MAE: {mae}')
print(f'MSE: {mse}')
