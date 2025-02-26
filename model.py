import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import adfuller


df = pd.read_csv("Grain_Prices_20250211.csv", parse_dates=['Date'], index_col='Date')
df = df.sort_index()  


print(df.info())
print(df.describe())


result = adfuller(df['Bid'])
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')


plot_acf(df['Bid'], lags=50)
plt.show()
plot_pacf(df['Bid'], lags=50)
plt.show()


df['Bid_diff'] = df['Bid'].diff().dropna()
result = adfuller(df['Bid_diff'].dropna())
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')


model = SARIMAX(df['Bid'],
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


df_reset = df.reset_index()  
bid_data = df_reset[['Date', 'Bid']]  
bid_data['Date'] = pd.to_datetime(bid_data['Date'])


df1 = bid_data.set_index('Date')
monthly_sales = df1.resample('M').mean()


plt.figure(figsize=(12, 6))
plt.plot(monthly_sales['Bid'], linewidth=3, c='purple')
plt.title("Grain Bids")
plt.xlabel("Date")
plt.ylabel("Bid")
plt.show()

forecast_mean = forecast.predicted_mean
forecast_conf_int = forecast.conf_int()

plt.figure(figsize=(12, 6))
plt.plot(monthly_sales['Bid'], label='Historical Data', linewidth=3, c='purple')
plt.plot(forecast_index, forecast_mean, label='Forecasted Data', linestyle='--', color='green', linewidth=3)
plt.fill_between(forecast_index, forecast_conf_int.iloc[:, 0], forecast_conf_int.iloc[:, 1], color='green', alpha=0.2)
plt.title("Grain Bids Forecast")
plt.xlabel("Date")
plt.ylabel("Bid")
plt.legend()
plt.show()

actuals = df['Bid'][-12:]  
forecasted = forecast_mean

mae = mean_absolute_error(actuals, forecasted)
mse = mean_squared_error(actuals, forecasted)
print(f'MAE: {mae}')
print(f'MSE: {mse}')
