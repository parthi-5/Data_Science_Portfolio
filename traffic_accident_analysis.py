import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# STEP 1: SET THE CLEAN VISUAL THEME
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'

# STEP 2: LOAD A SAFE CHUNK OF THE DATA
try:
    df = pd.read_csv('data/US_Accidents_March23.csv', nrows=50000)
except FileNotFoundError:
    print("Error: Could not find the US Accidents CSV file. Please check the name!")
    exit()

# STEP 3: CLEAN DATA & EXTRACT TIME DETAILS
df['Weather_Condition'] = df['Weather_Condition'].fillna('Clear')

df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')

df['Hour'] = df['Start_Time'].dt.hour

df = df.dropna(subset=['Hour'])

# STEP 4: SAVE THE CLEANED DATASET
df.to_csv('cleaned_traffic_accidents.csv', index=False)
print("Success: 'cleaned_traffic_accidents.csv' created and saved to your folder!")

# STEP 5: PLOT 1 - TOP 10 WEATHER CONDITIONS
fig1, ax1 = plt.subplots(figsize=(10, 5))

top_weather = df['Weather_Condition'].value_counts().head(10)

sns.barplot(x=top_weather.values, y=top_weather.index, palette='Reds_r', ax=ax1)

sns.despine(left=True, bottom=True)
ax1.set_title('Top 10 Weather Conditions During Accidents', fontsize=14, fontweight='bold', pad=15)
ax1.set_xlabel('Number of Recorded Accidents', fontsize=11, fontweight='semibold')

plt.tight_layout()
plt.savefig('accident_weather_factors.png', dpi=300)
plt.show()

# STEP 6: PLOT 2 - TIME OF DAY TREND (Hourly)
fig2, ax2 = plt.subplots(figsize=(10, 5))

hourly_accidents = df['Hour'].value_counts().sort_index()

plt.plot(hourly_accidents.index, hourly_accidents.values, marker='o', color='#E74C3C', linewidth=2.5)

sns.despine(left=True, bottom=True)
ax2.set_title('Traffic Accident Distribution by Hour of Day', fontsize=14, fontweight='bold', pad=15)
ax2.set_xlabel('Hour of Day (00:00 to 23:00)', fontsize=11, fontweight='semibold')
ax2.set_ylabel('Number of Accidents', fontsize=11, fontweight='semibold')
plt.xticks(range(0, 24)) 

plt.tight_layout()
plt.savefig('accident_time_distribution.png', dpi=300)
plt.show()