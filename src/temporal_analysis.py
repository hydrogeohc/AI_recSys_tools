import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt

# Generate sample flight time data (similar to the image)
np.random.seed(42)
dates = pd.date_range(start='2020-12-29', end='2021-05-04', freq='D')
flights = np.random.normal(loc=100, scale=25, size=len(dates))
flights = np.clip(flights, 20, 200)  # Clip values to match the range in the image

# Create DataFrame
data = pd.DataFrame({'ds': dates, 'y': flights})

# Define and fit the model
model = Prophet()
model.fit(data)

# Forecast future flights
future = model.make_future_dataframe(periods=30)  # Forecast for the next 30 days
forecast = model.predict(future)

# Visualize the forecast
fig = model.plot(forecast)
plt.title('Flight Time Temporal Analysis and Forecast')
plt.xlabel('Date')
plt.ylabel('Flight Time')
plt.show()

# Print the forecasted data for the next 5 days
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# Plot components of the forecast
fig2 = model.plot_components(forecast)
plt.show()