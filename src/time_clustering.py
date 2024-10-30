import numpy as np
import pandas as pd
from tslearn.clustering import TimeSeriesKMeans
import matplotlib.pyplot as plt

# Generate sample time series data
np.random.seed(42)
dates = pd.date_range(start='2020-12-29', end='2021-05-04', freq='D')
num_series = 5
data = []

for _ in range(num_series):
    series = np.random.normal(loc=100, scale=25, size=len(dates))
    series = np.clip(series, 20, 200)
    data.append(series)

# Convert to numpy array for clustering
data = np.array(data)

# Perform time series clustering
n_clusters = 3
model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", max_iter=10)
labels = model.fit_predict(data)

# Plot the clustered time series
plt.figure(figsize=(10, 6))
for i in range(n_clusters):
    plt.subplot(n_clusters, 1, i + 1)
    for series in data[labels == i]:
        plt.plot(dates, series)
    plt.title(f'Cluster {i + 1}')
plt.tight_layout()
plt.show()

print("Cluster labels:", labels)