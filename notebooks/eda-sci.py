import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.gofplots import qqplot
from pyod.models.iforest import IForest

# %% Setup Plotly
config = {'scrollZoom': True}
original_show = go.Figure.show

ratsoenrotraitodef custom_show(self, *args, **kwargs):
    original_show(self, *args, **kwargs)

go.Figure.show = custom_show

# %% Load daily snow values and preprocess
df = pd.read_csv('./data/daily_snow.csv', index_col='measure_date', parse_dates=True)

df.info()
df.isnull().sum()
df = df[df['HS'].notna()]

df['month'] = df.index.month
df['day'] = df.index.day
df['day_of_week'] = df.index.dayofweek
df['day_of_month'] = df.index.days_in_month
df['season'] = df['month'].apply(lambda x: (x % 12 + 3) // 3)

df.rename(columns={'hyear': 'year'}, inplace=True)

df.sample()

# %% Create sample
station = df['2015-01-01':].groupby('station_code').count().sort_values(by='HS', ascending=False).index[0]
sample = df[df['station_code'] == station].drop(columns=['station_code'])

# %% Plot sample station
sample.plot(y='HS', use_index=True)
plt.show()

# Plot different variables
sample.plot(subplots=True, figsize=(6, 6))
plt.show()

# Percent change
(sample['HS'].pct_change(1) * 100).plot(use_index=True, legend=True)
plt.show()

# %% Show sample using plotly
fig = go.Figure()
fig.add_trace(go.Scatter(x=sample.index, y=sample['HS'], name='HS'))
fig.show(config=config)

# %% STL Decomposition
stl = seasonal_decompose(sample['HS'], model='additive', period=365)
# Plot components
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(10, 8))

ax1.plot(stl.observed)
ax1.set_title('Observed')

ax2.plot(stl.trend)
ax2.set_title('Trend')

ax3.plot(stl.seasonal)
ax3.set_title('Seasonal')

ax4.plot(stl.resid)
ax4.set_title('Residual')

plt.tight_layout()
plt.show()

# %% Auto ARIMA
weekly_sample = sample['HS'].resample('M').mean()
train_data = weekly_sample[:-12]
model = auto_arima(weekly_sample, trace=True, suppress_warnings=True, seasonal=True, m=12)
model.summary()

predictions = model.predict(n_periods=24)

# Create a time index for the predictions
predicted_index = pd.date_range(start=train_data.index[-1] + pd.Timedelta(days=1), periods=12, freq='D')

# Plot actual vs predicted values
plt.figure(
    figsize=(20, 6),
    dpi=80,
    facecolor='w',
    edgecolor='k'
)
plt.plot(weekly_sample.index, weekly_sample, label='Actual')
plt.plot(weekly_sample.index[-24:], predictions, label='Predicted')
plt.legend()
plt.show()

# Calculate RMSE
data_range = train_data.max() - train_data.min()

rmse = mean_squared_error(weekly_sample[-24:], predictions, squared=False)
print(f'RMSE: {rmse}')

# %% EDA
sample.describe()
n_bins = int(np.sqrt(len(sample)))

# Histogram
plt.figure()
sample['HS'].hist(bins=n_bins, figsize=(20, 10))
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

# Normalize data for daily_snow_values on the HS column using min-max scaling
def normalize(df, feature_name='HS'):
    # use quantile transformer
    result = df.copy()
    max_value = df[feature_name].max()
    min_value = df[feature_name].min()
    result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


# Show normal and normalized boxplot
_, ax = plt.subplots(1, 2, figsize=(20, 10))

ax[0].boxplot(df['HS'])
ax[0].set_title('Daily snow values')

ax[1].boxplot(normalize(df[['HS']])['HS'])
ax[0].set_title('Normalizesd Daily snow values')

plt.tight_layout()
plt.show()

# show distribution of data using normal probability plot
plt.figure()
qqplot(sample['HS'], line='s')
plt.show()


# %% Show timeseries for top outliers
def analyze_outliers(df, column='HS', top_n=5, iqr_scope='global', plot=False):
    def get_iqr(data):
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        return Q3 - Q1, Q1, Q3

    def count_outliers(data, lower_bound, upper_bound):
        return ((data < lower_bound) | (data > upper_bound)).sum()

    if iqr_scope == 'global':
        global_iqr, global_Q1, global_Q3 = get_iqr(df[column])

    def get_outlier_counts(group):
        if iqr_scope == 'local':
            local_iqr, local_Q1, local_Q3 = get_iqr(group[column])
            lower_bound = local_Q1 - 1.5 * local_iqr
            upper_bound = local_Q3 + 1.5 * local_iqr
        else:
            lower_bound = global_Q1 - 1.5 * global_iqr
            upper_bound = global_Q3 + 1.5 * global_iqr
        return count_outliers(group[column], lower_bound, upper_bound)

    outlier_counts = df.groupby('station_code').apply(get_outlier_counts)
    top_outlier_stations = outlier_counts.nlargest(top_n)
    station_codes = top_outlier_stations.index

    if plot:
        fig, axs = plt.subplots(nrows=top_n, figsize=(10, 5 * top_n))
        for i, station_code in enumerate(station_codes):
            station_data = df[df['station_code'] == station_code]
            station_data.plot(y=column, use_index=True, ax=axs[i], title=f'Station {station_code}')
        plt.tight_layout()
        plt.show()

    return top_outlier_stations

top_stations = analyze_outliers(df, plot=True, iqr_scope='local')


# %% IForest
# station_name = 'ZNZ2'
station_name = 'SPN2'

single_station = df[df['station_code'] == station_name]

# Fit the Isolation Forest model
iforest = IForest()
iforest.fit(df.drop(columns=['station_code', 'HN_1D']).values)

# Predict outlier probabilities
probs = iforest.predict_proba(single_station.drop(columns=['station_code', 'HN_1D']).values)

# Identify outliers
outliers = single_station.iloc[probs[:, 1] >= 0.80]

# Display the time series with marked outliers
fig = go.Figure()
fig.add_trace(go.Scatter(x=single_station.index, y=single_station['HS'], name='HS'))
fig.add_trace(go.Scatter(x=outliers.index, y=outliers['HS'], mode='markers', marker=dict(color='red'), name='Outliers'))
fig.show()


# Fit LSTM
# %% Setup