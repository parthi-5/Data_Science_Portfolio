import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="My Data Science Portfolio", layout="wide")
sns.set_theme(style="white")
plt.rcParams['font.family'] = 'sans-serif'

st.sidebar.title("🧭 Navigation Hub")
page = st.sidebar.radio("Explore Analytics Hub:", [
    "Portfolio Executive Summary", 
    "Case Study 1: Global Demographics", 
    "Case Study 2: Titanic Survival EDA", 
    "Pipeline 3: Purchase Behavior ML Model", 
    "Pipeline 4: US Traffic Safety Analytics"
])


if page == "Portfolio Executive Summary":
    st.title("🚀 My Data Science Project Portfolio")
    st.write("Welcome! This central dashboard holds the analytical systems and machine learning tasks I engineered.")
    st.markdown("---")
    st.subheader("💡 Portfolio Architecture")
    st.write("Navigate through the sidebar to explore live interactive data renderings, visual insights, and exportable data segments for each specific case study.")

elif page == "Case Study 1: Global Demographics":
    st.title("📊 Case Study 1: Global Population Distribution")
    st.write("Analyzing country distribution scales using modern logarithmic visualizations to eliminate massive regional skewing.")
    
    try:
        df = pd.read_csv('data/API_SP.POP.TOTL_DS2_en_csv_v2_406129.csv', skiprows=4)
        non_countries = ['WLD', 'IBD', 'IBT', 'LMY', 'MIC', 'IBRD', 'TEA', 'HIC', 'LAC', 'ECA', 'SAS', 'SSF']
        df_countries = df[~df['Country Code'].isin(non_countries)].dropna(subset=['2023']).copy()
        df_countries['Pop_Millions'] = df_countries['2023'] / 1_000_000

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data=df_countries, x='Pop_Millions', bins=25, kde=True, log_scale=True, color='#4A90E2', ax=ax)
        
        median_pop = df_countries['Pop_Millions'].median()
        ax.axvline(median_pop, color='#E67E22', linestyle=':', linewidth=2, label=f'Median: {median_pop:.1f}M')
        
        sns.despine(left=True, bottom=True)
        ax.set_title('Global Population Logarithmic Distribution Profile (2023)', fontweight='bold')
        plt.legend()
        st.pyplot(fig)
        
    except FileNotFoundError:
        st.warning("Please verify 'API_SP.POP.TOTL_DS2_en_csv_v2_406129.csv' exists in your root folder path.")

elif page == "Case Study 2: Titanic Survival EDA":
    st.title("🚢 Case Study 2: Titanic Survival Exploratory Data Analysis")
    st.write("Uncovering systematic data patterns relative to socioeconomic categories and passenger demographics.")
    
    try:
        df = pd.read_csv('data/train.csv')
        df['Age'] = df['Age'].fillna(df['Age'].mean())
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
        
        st.subheader("1. Survival Distribution by Gender")
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        sns.barplot(data=df, x='Sex', y='Survived', palette={'male': '#4A90E2', 'female': '#FF6B6B'}, errorbar=None, ax=ax1)
        sns.despine(left=True, bottom=True)
        for bar in ax1.patches:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{bar.get_height():.1%}', ha='center', fontweight='bold')
        st.pyplot(fig1)
        
        st.subheader("2. Survival Distribution by Passenger Age")
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        sns.histplot(data=df, x='Age', hue='Survived', multiple='stack', bins=25, palette={1: '#2ECC71', 0: '#E74C3C'}, ax=ax2)
        sns.despine(left=True, bottom=True)
        plt.legend(title='Outcome', labels=['Survived', 'Did Not Survive'], frameon=False)
        st.pyplot(fig2)
        
        st.markdown("---")
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 Download Cleaned Titanic Dataset (CSV)", data=csv_data, file_name="cleaned_titanic_passenger_data.csv", mime="text/csv")
        
    except FileNotFoundError:
        st.warning("Please verify 'train.csv' is saved in your root directory folder.")

elif page == "Pipeline 3: Purchase Behavior ML Model":
    st.title("🎯 Pipeline 3: Consumer Purchase Prediction Pipeline")
    st.write("Supervised tree classifier optimizing numeric customer matrices to forecast target purchase flags.")
    
    try:
        df = pd.read_csv('data/bank-full.csv', sep=';')
        df['y'] = df['y'].map({'yes': 1, 'no': 0})
        numeric_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
        
        X = df[numeric_features]
        y = df['y']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = DecisionTreeClassifier(max_depth=4, random_state=42)
        model.fit(X_train, y_train)
        
        st.subheader("1. Predictive Driver Ranking Weight")
        importances = model.feature_importances_
        feature_ranking = pd.Series(importances, index=numeric_features).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(x=feature_ranking.values, y=feature_ranking.index, palette='viridis', ax=ax)
        sns.despine(left=True, bottom=True)
        st.pyplot(fig)
        
        # DOWNLOAD BUTTON
        st.markdown("---")
        csv_bank = X.copy()
        csv_bank['Target_Purchased'] = y
        st.download_button(label="📥 Download Encoded Numeric Marketing Features (CSV)", data=csv_bank.to_csv(index=False).encode('utf-8'), file_name="cleaned_numeric_bank_marketing.csv", mime="text/csv")
        
    except FileNotFoundError:
        st.warning("Please verify 'bank-full.csv' is present inside the workspace directory folder.")

elif page == "Pipeline 4: US Traffic Safety Analytics":
    st.title("🚗 Pipeline 4: US Traffic Accident Pattern Analytics")
    st.write("Evaluating continuous environmental lighting and time patterns across localized traffic accident data layers.")
    
    try:
        df = pd.read_csv('data/US_Accidents_March23.csv', nrows=50000)
        df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
        df['Hour'] = df['Start_Time'].dt.hour
        df = df.dropna(subset=['Hour'])

        st.subheader("1. Atmospheric Weather Distribution Factors")
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        top_weather = df['Weather_Condition'].value_counts().head(10)
        sns.barplot(x=top_weather.values, y=top_weather.index, palette='Reds_r', ax=ax1)
        sns.despine(left=True, bottom=True)
        st.pyplot(fig1)

        st.subheader("2. Hourly Incident Timeline Trends (Rush Hour Analysis)")
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        hourly_counts = df['Hour'].value_counts().sort_index()
        plt.plot(hourly_counts.index, hourly_counts.values, marker='o', color='#E74C3C', linewidth=2)
        sns.despine(left=True, bottom=True)
        plt.xticks(range(0, 24))
        st.pyplot(fig2)
        
        st.markdown("---")
        st.download_button(label="📥 Download Processed Accident Matrix Data (CSV)", data=df.to_csv(index=False).encode('utf-8'), file_name="processed_traffic_accidents.csv", mime="text/csv")
        
    except FileNotFoundError:
        st.warning("Please run your data generator script to initialize 'US_Accidents_March23.csv' inside this directory.")