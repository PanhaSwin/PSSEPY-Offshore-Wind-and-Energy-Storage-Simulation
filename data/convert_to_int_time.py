import pandas as pd

# --- Load the Excel File ---
df = pd.read_excel("generation_data.xlsx", index_col='Date', parse_dates=True)

# --- Define Biweekly Periods ---
# Every 14 days starting from the first date
start_date = df.index.min().normalize()
biweekly_index = ((df.index - start_date).days // 14).astype(int)
df['biweekly_id'] = biweekly_index

# --- Aggregate per Biweekly Group ---
# You can choose .mean(), .sum(), .first(), etc.
grouped = df.groupby('biweekly_id').agg({
    'DCT_Biweekly_Mean_MW': 'mean',
    'Biweekly_Kalman_Mean_MW': 'mean',
    'Kalman_Trend_MW_per_day': 'mean'  # or use 'first' to keep early trends
})

# --- Optional: Add start date of each biweekly window ---
grouped['biweekly_start'] = [start_date + pd.Timedelta(days=14 * i) for i in grouped.index]

# --- Reorder columns ---
grouped = grouped[['biweekly_start', 'DCT_Biweekly_Mean_MW', 'Biweekly_Kalman_Mean_MW', 'Kalman_Trend_MW_per_day']]

# --- Export to Excel ---
grouped.to_excel("generation_biweekly_grouped.xlsx", index_label='biweekly_id')

print("✅ Biweekly grouping complete! Output saved as 'generation_biweekly_grouped.xlsx'.")


import matplotlib.pyplot as plt

# --- Plot Biweekly Aggregates ---
fig, ax1 = plt.subplots(figsize=(14, 6))

# Power outputs (left y-axis)
ax1.plot(grouped['biweekly_start'], grouped['DCT_Biweekly_Mean_MW'], label="DCT Mean", color='green', linewidth=2)
ax1.plot(grouped['biweekly_start'], grouped['Biweekly_Kalman_Mean_MW'], label="Kalman Mean", color='blue', linestyle='--')
ax1.set_ylabel("Power Output (MW)")
ax1.set_xlabel("Biweekly Start Date")
ax1.tick_params(axis='y')
ax1.grid(True)

# --- Force all x-axis labels to show ---
ax1.set_xticks(grouped['biweekly_start'])  # every biweekly start
ax1.set_xticklabels(grouped['biweekly_start'].dt.strftime('%Y-%m-%d'), rotation=45, ha='right')


# Trend values (right y-axis)
ax2 = ax1.twinx()
ax2.bar(grouped['biweekly_start'], grouped['Kalman_Trend_MW_per_day'], width=10, alpha=0.4,
        color='purple', label="Trend (MW/day)")
ax2.set_ylabel("Trend (MW/day)", color='purple')
ax2.tick_params(axis='y', labelcolor='purple')

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.title("Biweekly Wind Power Aggregates and Trend Validation")
plt.tight_layout()
plt.show()
