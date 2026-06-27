import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_theme(style="white") 
plt.rcParams['font.family'] = 'sans-serif'


try:
    df = pd.read_csv('data/train.csv')
except FileNotFoundError:
    print("Error: Could not find 'train.csv'. Make sure it is in this folder!")
    exit()



df['Age'] = df['Age'].fillna(df['Age'].mean())           
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0]) 

df = df.drop(columns=['Cabin', 'PassengerId', 'Name', 'Ticket'])


fig1, ax1 = plt.subplots(figsize=(8, 5))

sns.barplot(
    data=df, 
    x='Sex', 
    y='Survived', 
    palette={'male': '#4A90E2', 'female': '#FF6B6B'}, 
    errorbar=None, 
    width=0.5,
    ax=ax1
)

sns.despine(left=True, bottom=True)
ax1.set_title('Survival Rates: Men vs Women on the Titanic', fontsize=14, fontweight='bold', pad=15)
ax1.set_xlabel('Gender', fontsize=11, fontweight='semibold')
ax1.set_ylabel('Survival Rate (Percentage)', fontsize=11, fontweight='semibold')

for bar in ax1.patches:
    height = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width()/2, 
        height + 0.02, 
        f'{height:.1%}', 
        ha='center', 
        fontweight='bold'
    )

plt.tight_layout()
plt.savefig('titanic_gender_pattern.png', dpi=300)
plt.show()

fig2, ax2 = plt.subplots(figsize=(9, 5))

sns.histplot(
    data=df,
    x='Age',
    hue='Survived',
    multiple='stack',      
    bins=25,               
    palette={1: '#2ECC71', 0: '#E74C3C'}, 
    edgecolor='white',
    alpha=0.8,
    ax=ax2
)

sns.despine(left=True, bottom=True)
ax2.set_title('Age Distribution of Survivors vs Deceased', fontsize=14, fontweight='bold', pad=15)
ax2.set_xlabel('Age of Passengers', fontsize=11, fontweight='semibold')
ax2.set_ylabel('Number of Passengers', fontsize=11, fontweight='semibold')

plt.legend(title='Outcome', labels=['Survived', 'Did Not Survive'], frameon=False)

df.to_csv('cleaned_titanic_train.csv', index=False)
print("Success: 'cleaned_titanic_train.csv' has been created and saved!")

plt.tight_layout()
plt.savefig('titanic_age_distribution.png', dpi=300)
plt.show()