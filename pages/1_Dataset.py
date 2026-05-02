import streamlit as st
import pandas as pd

st.set_page_config(page_title="Dataset - Used Car Price Prediction", layout="wide")

st.title("📂 Dataset Explorer")
st.markdown("This page display raw data that is used for model training.")

@st.cache_data
def load_raw_data():
    try:
        df1 = pd.read_csv("dataset/used_car.csv")
        df2 = pd.read_csv("dataset/used_car_data_new.csv")
        return df1, df2
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None, None

df1_raw, df2_raw = load_raw_data()

if df1_raw is not None and df2_raw is not None:
    tab1, tab2 = st.tabs(["Source 1", "Source 2"])
    
    with tab1:
        st.header("used_car.csv")
        st.subheader("Raw Data")
        st.dataframe(df1_raw, use_container_width=True)
        st.write(f"Total Row: **{df1_raw.shape[0]}**, Total Column: **{df1_raw.shape[1]}**")
        
    with tab2:
        st.header("used_car_data_new.csv")
        st.subheader("Raw Data")
        st.dataframe(df2_raw, use_container_width=True)
        st.write(f"Total Row: **{df2_raw.shape[0]}**, Total Column: **{df2_raw.shape[1]}**")

    st.divider()
    st.subheader("Descriptive Statistics")
    st.write("Overview of numerical data distribution:")
    st.write(df1_raw.describe())
else:
    st.warning("Make sure the 'dataset/' folder contains the files 'used_car.csv' and 'used_car_data_new.csv'.")