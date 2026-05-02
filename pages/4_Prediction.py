import streamlit as st
import pandas as pd
import numpy as np
import joblib
import difflib

# Page Configuration
st.set_page_config(page_title="Price Prediction - Used Car", layout="wide")

# --- SUPPORTING FUNCTIONS ---
@st.cache_resource
def load_models():
    try:
        model = joblib.load("model/model.pkl")
        le_trans = joblib.load("model/le_trans.pkl")
        model_columns = joblib.load("model/model_columns.pkl")
        return model, le_trans, model_columns
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return None, None, None

@st.cache_data
def load_prediction_data():
    # Load raw data for valid brand/model lists
    df1 = pd.read_csv("dataset/used_car.csv")
    df2 = pd.read_csv("dataset/used_car_data_new.csv")
    df2['transmission'] = df2['id_transmission'].map({1: 'manual', 2: 'automatic'})
    df = pd.concat([df1, df2], ignore_index=True)
    
    df['brand'] = df['brand'].astype(str).str.lower().str.strip()
    df['model'] = df['model'].astype(str).str.lower().str.strip()
    
    valid_brands = df['brand'].unique()
    brand_model_map = df.groupby('brand')['model'].unique().to_dict()
    return valid_brands, brand_model_map

def smart_match(input_str, choices, threshold=0.5):
    input_str = input_str.lower().strip()
    if input_str in choices:
        return input_str, 1.0
    best_match = None
    best_score = 0
    for c in choices:
        score = difflib.SequenceMatcher(None, input_str, c).ratio()
        if score > best_score:
            best_score = score
            best_match = c
    return best_match if best_score >= threshold else None, best_score

# --- PREDICTION UI ---
st.title("🚗 Car Price Predictor")
st.write("Enter the vehicle details below to get a market price estimate.")

model_rf, le_trans, model_columns = load_models()
valid_brands, brand_model_map = load_prediction_data()

if model_rf is not None:
    with st.container(border=True):
        col1, col2 = st.columns(2)
        
        with col1:
            brand_input = st.text_input("Car Brand", placeholder="e.g., Toyota, Honda...")
            model_input = st.text_input("Car Model", placeholder="e.g., Avanza, Brio...")
            transmission = st.selectbox("Transmission", options=le_trans.classes_)
            
        with col2:
            year = st.number_input("Vehicle Year", 2000, 2026, 2020)
            mileage = st.number_input("Mileage (KM)", 0, 1000000, 50000, step=1000)

    if st.button("Calculate Estimated Price", type="primary", use_container_width=True):
        if not brand_input or not model_input:
            st.warning("Please fill in the brand and model.")
        else:
            # BRAND MATCH
            brand_match, b_score = smart_match(brand_input, valid_brands)
            if not brand_match:
                st.error(f"Brand '{brand_input}' not found in database.")
                st.stop()

            # MODEL MATCH
            filtered_models = brand_model_map.get(brand_match, [])
            model_match, m_score = smart_match(model_input, filtered_models)
            
            if not model_match:
                st.error(f"Model '{model_input}' is not registered for {brand_match.title()}.")
                st.stop()

            # PREPARATION & PREDICTION
            st.info(f"Detecting: **{brand_match.title()} {model_match.title()}** ({transmission})")
            
            # Feature Engineering
            car_age = 2026 - year
            input_df = pd.DataFrame(0, index=[0], columns=model_columns)
            input_df['mileage (km)'] = mileage
            input_df['car_age'] = car_age
            input_df['transmission'] = le_trans.transform([transmission])[0]
            
            # One-Hot Encoding Sets
            if f"brand_{brand_match}" in model_columns:
                input_df[f"brand_{brand_match}"] = 1
            if f"model_{model_match}" in model_columns:
                input_df[f"model_{model_match}"] = 1
            
            # Predict
            pred_log = model_rf.predict(input_df)[0]
            pred_price = np.expm1(np.clip(pred_log, 0, 20))

            # RESULT DISPLAY
            st.success(f"### Estimated Price: Rp {int(pred_price):,}")
            st.caption("This price is an estimate based on market data. Physical condition can affect the final value.")

else:
    st.error("Prediction tool failed to run because models were not found.")