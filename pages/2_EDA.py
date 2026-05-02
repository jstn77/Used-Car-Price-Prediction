import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Konfigurasi Halaman
st.set_page_config(page_title="EDA - Used Car Price Prediction", layout="wide")

st.title("📊 Exploratory Data Analysis")
st.markdown("Visual analysis to understand the characteristics of used car data in Indonesia.")

# Fungsi Load & Process Data (Sama dengan app.py)
@st.cache_data
def load_processed_data():
    try:
        df1 = pd.read_csv("dataset/used_car.csv")
        df2 = pd.read_csv("dataset/used_car_data_new.csv")
        
        # Preprocessing sederhana untuk EDA
        df2['transmission'] = df2['id_transmission'].map({1: 'manual', 2: 'automatic'})
        df1['mileage (km)'] = df1['mileage (km)'] * 1000
        
        df = pd.concat([df1, df2], ignore_index=True)
        df = df[['brand', 'model', 'year', 'mileage (km)', 'transmission', 'price (Rp)']]
        
        # Normalisasi teks
        for col in ['brand', 'model', 'transmission']:
            df[col] = df[col].astype(str).str.lower().str.strip()
            
        return df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None

df = load_processed_data()

if df is not None:
    # --- BAGIAN 1: OVERVIEW STATISTIK ---
    st.header("1. Data Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Row", df.shape[0])
    with col2:
        st.metric("Total Brand", df['brand'].nunique())
    with col3:
        st.metric("Price Average", f"Rp {df['price (Rp)'].mean():,.0f}")

    st.write("### Descriptive Statistics")
    st.write(df.describe())

    st.divider()

    # --- BAGIAN 2: DISTRIBUSI & KATEGORI ---
    st.header("2. Distribution & Composition")
    
    tab1, tab2, tab3 = st.tabs(["Price Distribution", "Top Brand", "Transmition"])
    
    with tab1:
        st.subheader("Car Price Distribution")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(df['price (Rp)'], bins=40, kde=True, ax=ax, color='skyblue')
        plt.xlabel("Price (Rp)")
        st.pyplot(fig)
        
    with tab2:
        st.subheader("Top 10 Brand in Dataset")
        fig, ax = plt.subplots(figsize=(10, 5))
        df['brand'].value_counts().head(10).plot(kind='bar', ax=ax, color='coral')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
    with tab3:
        st.subheader("Comparison of Transmission Types")
        fig, ax = plt.subplots(figsize=(8, 5))
        df['transmission'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax, startangle=90)
        ax.set_ylabel("")
        st.pyplot(fig)

    st.divider()

    # --- BAGIAN 3: HUBUNGAN ANTAR VARIABEL ---
    st.header("3. Correlation Analysis")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Mileage vs Price")
        fig, ax = plt.subplots()
        sns.scatterplot(x=df['mileage (km)'], y=df['price (Rp)'], alpha=0.5, ax=ax)
        st.pyplot(fig)
        st.caption("The higher the mileage, the more it tends to lower the price.")
        
    with col_b:
        st.subheader("Car Year vs Price")
        fig, ax = plt.subplots()
        sns.scatterplot(x=df['year'], y=df['price (Rp)'], alpha=0.5, color='green', ax=ax)
        st.pyplot(fig)
        st.caption("Newer year cars have a higher price range.")

else:
    st.warning("Data not available. Please check your dataset folder.")