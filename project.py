import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Load and group data by State
df = pd.read_csv('indian_cities_raw_aqi_dataset_133cities_2016_2025.csv')
years = [str(y) for y in range(2016, 2026)]

state_df = df.groupby('State')[years].mean()

# Sort states from highest 10-year average to lowest
state_df['Mean'] = state_df.mean(axis=1)
state_df = state_df.sort_values(by='Mean', ascending=False).drop(columns='Mean')

# 2. Create the heatmap
fig, ax = plt.subplots(figsize=(12, 11))
im = ax.imshow(state_df.values, cmap='YlOrRd', aspect='auto')

# 3. Set axis ticks and labels
ax.set_xticks(np.arange(len(years)))
ax.set_yticks(np.arange(len(state_df.index)))
ax.set_xticklabels(years, fontsize=10)
ax.set_yticklabels(state_df.index, fontsize=10)

# 4. Write the AQI value inside each cell
for i in range(len(state_df.index)):
    for j in range(len(years)):
        val = state_df.values[i, j]
        text_color = 'white' if val > 165 else 'black'
        ax.text(j, i, f'{val:.0f}', ha='center', va='center', color=text_color, fontsize=8.5)

# 5. Add colorbar and title
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Mean AQI', fontsize=11)
ax.set_title('State-Wise Annual AQI Heatmap (2016–2025)', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()

plt.show()




plt.figure(figsize=(12, 6))

# Plot all 133 cities in faint gray
for _, row in df.iterrows():
    plt.plot(years, row[years].values, color='lightgray', linewidth=0.8, alpha=0.6)

# Highlight specific cities on top
highlight_cities = ['Ghaziabad', 'Delhi', 'Mumbai', 'Bengaluru', 'Aizawl']
colors = ['#d62728', '#ff7f0e', '#1f77b4', '#2ca02c', '#9467bd']

for city, color in zip(highlight_cities, colors):
    city_row = df[df['City'] == city].iloc[0]
    plt.plot(years, city_row[years].values, marker='o', linewidth=2.5, color=color, label=city)

plt.title('10-Year AQI Trajectory: All 133 Cities vs. Selected Cities', fontsize=14, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Mean AQI', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend(title='Highlighted Cities', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()

plt.show()



import pandas as pd

# 1. Load the dataset
df = pd.read_csv('indian_cities_raw_aqi_dataset_133cities_2016_2025.csv')
years = [str(y) for y in range(2016, 2026)]

print("=" * 60)
print("1. DATASET OVERVIEW")
print("=" * 60)
print(f"Total Cities Covered : {df['City'].nunique()}")
print(f"Total States / UTs   : {df['State'].nunique()}")
print(f"Total Regions        : {df['Region'].nunique()}")
print(f"Missing Values       : {df.isnull().sum().sum()}")
print("\nCategory Breakdown:")
print(df['Overall_Category'].value_counts().to_string())

print("\n" + "=" * 60)
print("2. STATISTICAL SUMMARY (2016–2025)")
print("=" * 60)
print(df[years + ['10Yr_Mean_AQI', '10Yr_Change_%']].describe().round(2).to_string())

print("\n" + "=" * 60)
print("3. TOP 5 MOST POLLUTED CITIES (10-Year Mean AQI)")
print("=" * 60)
print(df[['City', 'State', '10Yr_Mean_AQI', '10Yr_Change_%', 'Overall_Category']].head(5).to_string(index=False))

print("\n" + "=" * 60)
print("4. TOP 5 CLEANEST CITIES (10-Year Mean AQI)")
print("=" * 60)
print(df[['City', 'State', '10Yr_Mean_AQI', '10Yr_Change_%', 'Overall_Category']].tail(5).to_string(index=False))

print("\n" + "=" * 60)
print("5. BIGGEST IMPROVERS VS. WORST DECLINES (2016 vs. 2025)")
print("=" * 60)
print("Most Improved Cities:")
print(df.sort_values('10Yr_Change_%')[['City', 'State', '2016', '2025', '10Yr_Change_%']].head(5).to_string(index=False))

print("\nCities with Worsening AQI (Positive % Change):")
worsened = df[df['10Yr_Change_%'] > 0].sort_values('10Yr_Change_%', ascending=False)
print(worsened[['City', 'State', '2016', '2025', '10Yr_Change_%']].to_string(index=False))

print("\n" + "=" * 60)
print("6. REGIONAL & STATE AVERAGES")
print("=" * 60)
region_summary = df.groupby('Region').agg(
    Cities=('City', 'count'),
    Mean_AQI=('10Yr_Mean_AQI', 'mean'),
    Avg_Change_Pct=('10Yr_Change_%', 'mean')
).sort_values('Mean_AQI', ascending=False).round(2)
print("Regional Summary:")
print(region_summary.to_string())

print("\nTop 5 States by Average AQI:")
state_summary = df.groupby('State')['10Yr_Mean_AQI'].mean().sort_values(ascending=False).round(2)
print(state_summary.head(5).to_string())
