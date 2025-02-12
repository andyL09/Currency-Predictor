print("wow")

#woah

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Generate sample data
np.random.seed(42)
dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
data = np.random.randn(100).cumsum() + 50  # Random walk for the sake of example
df = pd.DataFrame(data, index=dates, columns=['Value'])

# Visualize the data
df.plot(figsize=(10, 6))
plt.title('Generated Time Series')
plt.show()

# Define SARIMAX Model
# Let's assume the model parameters (p, d, q) = (1, 1, 1) and seasonal (P, D, Q, S) = (1, 1, 1, 7)
sarimax_model = SARIMAX(df['Value'],
                        order=(1, 1, 1),  # AR, I, MA order
                        seasonal_order=(1, 1, 1, 7),  # Seasonal AR, I, MA order, period = 7 (weekly seasonality)
                        enforce_stationarity=False,
                        enforce_invertibility=False)

# Fit the model
sarimax_fit = sarimax_model.fit(disp=False)

# Summary of the model
print(sarimax_fit.summary())

# Make predictions (in-sample and out-of-sample)
forecast_steps = 30  # Forecast for the next 30 days
predictions = sarimax_fit.get_forecast(steps=forecast_steps)
forecast_values = predictions.predicted_mean

# Confidence intervals for the forecast
conf_int = predictions.conf_int()

# Plot the forecast
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Value'], label='Historical Data')
plt.plot(pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='D'),
         forecast_values, color='red', label='Forecast')
plt.fill_between(pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='D'),
                 conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3)
plt.legend()
plt.title('SARIMAX Forecast')
plt.show()