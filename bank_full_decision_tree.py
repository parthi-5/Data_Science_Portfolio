import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# STEP 1: SET THE CLEAN BACKGROUND THEME
sns.set_theme(style="white")
plt.rcParams['font.family'] = 'sans-serif'

# STEP 2: LOAD THE MASTER DATASET (bank-full.csv)
# Loading the entire master dataset of 45,211 rows!
try:
    df = pd.read_csv('data/bank-full.csv', sep=';')
except FileNotFoundError:
    print("Error: Could not find 'bank-full.csv' in this folder!")
    exit()

# STEP 3: PREPARE THE DATA (Simple Method)
# Change the target column 'y' from words ('yes'/'no') to numbers (1/0)
df['y'] = df['y'].map({'yes': 1, 'no': 0})

numeric_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
X = df[numeric_features] 
y = df['y']              

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# STEP 4: TRAIN THE DECISION TREE MODEL
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

# STEP 5: CHECK MODEL ACCURACY
predictions = model.predict(X_test)
score = accuracy_score(y_test, predictions)

print("--- PROCESS COMPLETE ---")
print(f"Master Dataset Row Count Processed: {len(df):,}")
print(f"Model successfully predicted behavior with {score:.2%} accuracy.")

cleaned_bank_data = X.copy()
cleaned_bank_data['Target_Purchased'] = y

cleaned_bank_data.to_csv('cleaned_numeric_bank_full.csv', index=False)
print("Success: 'cleaned_numeric_bank_full.csv' has been created and saved!")

# STEP 6: CREATE FEATURE IMPORTANCE GRAPH
importances = model.feature_importances_
feature_ranking = pd.Series(importances, index=numeric_features).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(x=feature_ranking.values, y=feature_ranking.index, palette='viridis', ax=ax)

# Clean up the graph visual style
sns.despine(left=True, bottom=True)
ax.set_title('Top Drivers of Purchases on Master Dataset (bank-full)', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Importance Predictive Score', fontsize=11, fontweight='semibold')

plt.tight_layout()
plt.savefig('bank_full_feature_importance.png', dpi=300)
plt.show()