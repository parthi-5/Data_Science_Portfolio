import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans']

try:
    df = pd.read_csv('data/API_SP.POP.TOTL_DS2_en_csv_v2_406129.csv', skiprows=4)
except FileNotFoundError:
    print("Please place the downloaded World Bank CSV in the same directory and update the filename!")
    exit()


target_year = '2023'

non_countries = [
    'WLD', 'IBD', 'IBT', 'LMY', 'MIC', 'IBRD', 'TEA', 'LTE', 'EAP', 'EAR', 
    'UMC', 'EAS', 'HIC', 'TLA', 'LAC', 'LCN', 'LMC', 'ECA', 'TEC', 'ECS', 
    'EAS', 'MEA', 'MNA', 'TMN', 'NAC', 'SAS', 'TSA', 'SSF', 'SSA', 'TSS', 'PRE'
]
df_countries = df[~df['Country Code'].isin(non_countries)].copy()

df_countries = df_countries.dropna(subset=[target_year])

df_countries['Pop_Millions'] = df_countries[target_year] / 1_000_000

fig, ax = plt.subplots(figsize=(12, 7))

sns.histplot(
    data=df_countries, 
    x='Pop_Millions', 
    bins=30, 
    kde=True,                  
    log_scale=True,            
    color='#4A90E2',           
    edgecolor='white', 
    linewidth=1.2,
    alpha=0.85
)


plt.title(f'Global Population Distribution Across Countries ({target_year})', 
          fontsize=16, fontweight='bold', pad=20, color='#333333')
plt.xlabel('Population (Millions) — Logarithmic Scale', fontsize=12, labelpad=10, fontweight='semibold')
plt.ylabel('Number of Countries', fontsize=12, labelpad=10, fontweight='semibold')


sns.despine(left=True, bottom=True) # Removes ugly black border boxes
ax.xaxis.grid(True, linestyle='--', alpha=0.6, color='#CCCCCC')
ax.yaxis.grid(True, linestyle='--', alpha=0.6, color='#CCCCCC')


import matplotlib.ticker as ticker
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))

median_pop = df_countries['Pop_Millions'].median()
ax.axvline(median_pop, color='#E67E22', linestyle=':', linewidth=2, label=f'Median: {median_pop:.1f}M')

max_pop_country = df_countries.loc[df_countries['Pop_Millions'].idxmax()]
ax.text(max_pop_country['Pop_Millions'] * 0.4, 3, f"Outliers\n(e.g., India/China\n>1.4 Billion)", 
        color='#C0392B', fontsize=10, weight='bold', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

plt.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none')

plt.tight_layout()

plt.savefig('world_population_distribution.png', dpi=300)
plt.show()