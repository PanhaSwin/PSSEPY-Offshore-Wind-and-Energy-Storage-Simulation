import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.fft import dct, idct
from scipy.stats import linregress
from pykalman import KalmanFilter

# --- Load Datasets ---
u_ds_2023 = xr.open_dataset("loc1-2023-100m-u.nc")
v_ds_2023 = xr.open_dataset("loc1-2023-100m-v.nc")
u_ds_2024 = xr.open_dataset("loc1-2024-100m-u.nc")
v_ds_2024 = xr.open_dataset("loc1-2024-100m-v.nc")

u100 = xr.concat([u_ds_2023['u100'], u_ds_2024['u100']], dim='valid_time')
v100 = xr.concat([v_ds_2023['v100'], v_ds_2024['v100']], dim='valid_time')
wind_speed = np.sqrt(u100**2 + v100**2)
mean_wind_speed = wind_speed.mean(dim=["latitude", "longitude"])

# --- Power Conversion ---
air_density = 1.225
rotor_area = 10000
efficiency = 0.4
rated_power = 100e6
raw_power = 0.5 * air_density * rotor_area * (mean_wind_speed**3) * efficiency

raw_power *= 50  # Scaling for 50 turbines

raw_power = xr.where(raw_power > rated_power, rated_power, raw_power)
raw_power = raw_power.interpolate_na(dim='valid_time')

# --- Convert to Pandas Series ---
ts = pd.Series(raw_power.values, index=pd.to_datetime(raw_power['valid_time'].values))
ts = ts.asfreq('D').interpolate()

# --- Biweekly Setup ---
biweekly_starts = pd.date_range(ts.index[0], ts.index[-1], freq='SME')  # Semi-month end
biweekly_centers = [s + pd.Timedelta(days=6.5) for s in biweekly_starts if s + pd.Timedelta(days=13) <= ts.index[-1]]

# --- DCT Biweekly Reference ---
def dct_lowpass(series, keep=30):
    x = series.values
    coeffs = dct(x, norm='ortho')
    coeffs[keep:] = 0
    smoothed = idct(coeffs, norm='ortho')
    return pd.Series(smoothed, index=series.index)

dct_smooth_series = dct_lowpass(ts, keep=30)
biweekly_means_dct = [dct_smooth_series[s:s + pd.Timedelta(days=13)].mean()
                      for s in biweekly_starts if s + pd.Timedelta(days=13) <= ts.index[-1]]
biweekly_series_dct = pd.Series(biweekly_means_dct, index=biweekly_centers)

# --- Kalman Filter ---
kf = KalmanFilter(
    transition_matrices=[[1, 1], [0, 1]],
    observation_matrices=[[1, 0]],
    initial_state_mean=[ts.iloc[0], 0],
    transition_covariance=[[1e3, 0], [0, 1e3]],
    observation_covariance=1e8
)

state_means, _ = kf.filter(ts.values)
kf_series = pd.Series(state_means[:, 0], index=ts.index)

# --- Causal Low-Pass Filter (IIR) for Trend Smoothing Only ---
alpha = 0.1  # Adjust for smoothness (0.1 = very smooth, 0.5 = responsive)
smoothed_for_trend = pd.Series(index=kf_series.index, dtype=float)
smoothed_for_trend.iloc[0] = kf_series.iloc[0]

for i in range(1, len(kf_series)):
    smoothed_for_trend.iloc[i] = alpha * kf_series.iloc[i] + (1 - alpha) * smoothed_for_trend.iloc[i - 1]

# --- Masked Biweekly Kalman Trends ---
biweekly_trends = []
biweekly_trend_times = []

window_days = 3  # slope window
x = np.arange(window_days)

for start in biweekly_starts:
    end = start + pd.Timedelta(days=window_days - 1)
    if end > kf_series.index[-1]:
        continue

    # Raw Kalman block
    raw_block = kf_series[start:end]
    # Smoothed Kalman block
    smooth_block = smoothed_for_trend[start:end]

    if raw_block.isna().any() or smooth_block.isna().any():
        continue

    raw_slope, *_ = linregress(x, raw_block.values)
    smooth_slope, *_ = linregress(x, smooth_block.values)

    # Mask: keep only if signs agree
    if np.sign(raw_slope) == np.sign(smooth_slope):
        final_slope = raw_slope
    else:
        final_slope = 0  # discard false slope

    biweekly_trends.append(final_slope)
    biweekly_trend_times.append(start + pd.Timedelta(days=window_days / 2))

trend_series = pd.Series(biweekly_trends, index=biweekly_trend_times)


# --- Meanic Kalman Prediction ---
meanic_prediction_series = pd.Series(index=ts.index, dtype=float)

for start in biweekly_starts:
    end = start + pd.Timedelta(days=13)
    if end > kf_series.index[-1]:
        continue
    preview = kf_series[start:start + pd.Timedelta(days=5)]
    if preview.isna().any():
        continue
    mean_val = preview.mean()
    meanic_prediction_series.loc[start:end] = mean_val

# --- Fill Missing Predictions ---
meanic_prediction_series_filled = meanic_prediction_series.copy()
meanic_prediction_series_filled.bfill(inplace=True)
meanic_prediction_series_filled.ffill(inplace=True)

# --- Biweekly Kalman Trends (First 6 Days, from smoothed signal) ---
biweekly_trends = []
biweekly_trend_times = []

for start in biweekly_starts:
    end = start + pd.Timedelta(days=5)
    if end > smoothed_for_trend.index[-1]:
        continue
    block = smoothed_for_trend[start:end]
    if block.isna().any():
        continue
    x = np.arange(len(block))
    y = block.values
    slope, _, _, _, _ = linregress(x, y)
    biweekly_trends.append(slope)
    biweekly_trend_times.append(start + pd.Timedelta(days=2))

trend_series = pd.Series(biweekly_trends, index=biweekly_trend_times)

# --- Plot Combined Figure ---
fig, axs = plt.subplots(2, 1, figsize=(14, 9), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

# --- Top Plot: Kalman vs Means ---
axs[0].plot(kf_series.index, kf_series / 1e6, label="Kalman Filtered (Daily)", color="blue", linestyle=":", alpha=0.6, linewidth=1)
axs[0].plot(smoothed_for_trend.index, smoothed_for_trend / 1e6, label="Smoothed Kalman (For Trend)", color="black", linestyle="--", linewidth=1.5)
axs[0].plot(biweekly_series_dct.index, biweekly_series_dct.values / 1e6, label="DCT Biweekly Mean (Centered, Line)", color="green", linewidth=2)
axs[0].bar(biweekly_series_dct.index, biweekly_series_dct.values / 1e6, width=10, align='center',
           color='lightgreen', alpha=0.6, label="DCT Biweekly Mean (Centered, Bar)")

for i, start in enumerate(biweekly_starts):
    end = start + pd.Timedelta(days=13)
    if end > meanic_prediction_series.index[-1]:
        continue
    block = meanic_prediction_series[start:end]
    if block.isna().any():
        continue
    axs[0].bar(block.index, block.values / 1e6, width=1.0, align='center',
               color='red', alpha=0.4, label="Meanic Kalman Prediction (First 6 Days)" if i == 0 else "")

axs[0].set_ylabel("Power Output (MW)")
axs[0].set_title("Meanic Kalman Biweekly Prediction vs DCT Reference")
axs[0].grid(True)
axs[0].legend()

# --- Bottom Plot: Trend Bars ---
axs[1].bar(trend_series.index, trend_series.values / 1e6, width=10, align='center', color='purple', alpha=0.7)
axs[1].axhline(0, color='black', linestyle='--', linewidth=1)
axs[1].set_ylabel("Trend (MW/day)")
axs[1].set_title("Biweekly Kalman Power Trend (First 6 Days)")
axs[1].grid(True)

plt.xlabel("Date")
plt.tight_layout()
plt.show()

# --- Prepare Data for Export ---
combined_df = pd.DataFrame({
    'DCT_Biweekly_Mean_MW': biweekly_series_dct / 1e6,
    'Biweekly_Kalman_Mean_MW': meanic_prediction_series_filled.resample('2W').mean() / 1e6,
    'Kalman_Trend_MW_per_day': trend_series.resample('2W').mean() / 1e6,

})

# --- Fix DCT Biweekly Series (Step Fill Across Days) ---
dct_filled = biweekly_series_dct.copy()
dct_filled = dct_filled.reindex(pd.date_range(dct_filled.index.min(), dct_filled.index.max(), freq='D'))
dct_filled.ffill(inplace=True)

# --- Normalize to match 00:00:00 datetime resolution ---
normalized_index = dct_filled.index.normalize()

# --- Align Other Series ---
kalman_mean_filled = meanic_prediction_series_filled.reindex(normalized_index).ffill() / 1e6
trend_filled = trend_series.reindex(normalized_index).ffill() / 1e6

# --- Combine ---
export_df = pd.DataFrame({
    'DCT_Biweekly_Mean_MW': dct_filled.values / 1e6,
    'Biweekly_Kalman_Mean_MW': kalman_mean_filled.values,
    'Kalman_Trend_MW_per_day': trend_filled.values
}, index=normalized_index)

export_df.index.name = 'Date'

# --- Export ---
export_df.to_excel("generation_data.xlsx")

# --- Plot for Validation ---
import matplotlib.pyplot as plt

fig, ax1 = plt.subplots(figsize=(14, 6))

# Power outputs (left y-axis)
ax1.plot(export_df.index, export_df['DCT_Biweekly_Mean_MW'], label="DCT Biweekly Mean", color='green', linewidth=2)
ax1.plot(export_df.index, export_df['Biweekly_Kalman_Mean_MW'], label="Kalman Biweekly Mean", color='blue', linestyle='--')
ax1.set_ylabel("Power Output (MW)", color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.grid(True)

# Trend values (right y-axis)
ax2 = ax1.twinx()
ax2.plot(export_df.index, export_df['Kalman_Trend_MW_per_day'], label="Kalman Trend (MW/day)", color='purple', linestyle=':', linewidth=1.5)
ax2.set_ylabel("Trend (MW/day)", color='purple')
ax2.tick_params(axis='y', labelcolor='purple')

# Legends and layout
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.title("Validation: Aligned Biweekly Metrics (DCT vs Kalman)")
plt.tight_layout()
plt.show()
