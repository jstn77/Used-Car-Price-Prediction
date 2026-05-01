import streamlit as st
import pandas as pd
import numpy as np
import joblib
import difflib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Used Car Price Prediction", layout="wide")
st.title("Used Car Price Prediction Indonesia")

# LOAD DATA
@st.cache_data
def load_raw_data():
    df1 = pd.read_csv("dataset/used_car.csv")
    df2 = pd.read_csv("dataset/used_car_data_new.csv")
    return df1, df2

@st.cache_data
def load_processed_data():
    df1, df2 = load_raw_data()

    df2['transmission'] = df2['id_transmission'].map({1: 'manual', 2: 'automatic'})
    df1['mileage (km)'] = df1['mileage (km)'] * 1000

    df = pd.concat([df1, df2], ignore_index=True)

    df = df[['brand', 'model', 'year', 'mileage (km)', 'transmission', 'price (Rp)']]

    #Preprocessing
    df['brand'] = df['brand'].astype(str).str.lower().str.strip()
    df['model'] = df['model'].astype(str).str.lower().str.strip()
    df['transmission'] = df['transmission'].astype(str).str.lower().str.strip()

    return df

df1_raw, df2_raw = load_raw_data()
df = load_processed_data()

# LOAD MODEL
model = joblib.load("model/model.pkl")
le_trans = joblib.load("model/le_trans.pkl")
model_columns = joblib.load("model/model_columns.pkl")

# SMART MATCH
def smart_match(input_str, choices, threshold=0.5):
    input_str = input_str.lower().strip()

    # 1. exact match
    if input_str in choices:
        return input_str, 1.0

    # 2. similarity scoring
    best_match = None
    best_score = 0

    for c in choices:
        score = difflib.SequenceMatcher(None, input_str, c).ratio()
        if score > best_score:
            best_score = score
            best_match = c

    # 3. threshold check
    if best_score >= threshold:
        return best_match, best_score
    else:
        return None, best_score

valid_brands = df['brand'].unique()
valid_models = df['model'].unique()

# mapping brand => model
brand_model_map = df.groupby('brand')['model'].unique().to_dict()

# SIDEBAR
menu = st.sidebar.radio("Menu", [
    "Dataset",
    "EDA",
    "Preprocessing",
    "Training",
    "Prediction"
])

# DATASET
if menu == "Dataset":
    st.subheader("Raw Dataset")

    st.write("### Dataset 1: used_car.csv (RAW)")
    st.dataframe(df1_raw.head(500))
    st.write("Total Data (Row, Column):", df1_raw.shape)

    st.divider()

    st.write("### Dataset 2: used_car_data_new.csv (RAW)")
    st.dataframe(df2_raw.head(500))
    st.write("Total Data (Row, Column):", df2_raw.shape)

# EDA
elif menu == "EDA":
    st.subheader("Exploratory Data Analysis")

    st.write("## 1. Overview Data")
    st.write("Jumlah Data:", df.shape)
    st.write("Tipe Data:")
    st.write(df.dtypes)

    st.write("## 2. Statistik Deskriptif")
    st.write(df.describe())

    st.write("## 3. Missing Values")
    st.write(df.isnull().sum())

    st.divider()

    # DISTRIBUSI HARGA
    st.write("## 4. Distribusi Harga")

    fig, ax = plt.subplots()
    sns.histplot(df['price (Rp)'], bins=40, kde=True, ax=ax)
    ax.set_title("Distribusi Harga Mobil")
    st.pyplot(fig)

    # TOP BRAND
    st.write("## 5. Top Brand")

    fig, ax = plt.subplots()
    df['brand'].value_counts().head(10).plot(kind='bar', ax=ax)
    ax.set_title("Top 10 Brand")
    st.pyplot(fig)

    # TRANSMISSION
    st.write("## 6. Transmission Distribution")

    fig, ax = plt.subplots()
    df['transmission'].value_counts().plot(kind='bar', ax=ax)
    ax.set_title("Manual vs Automatic")
    st.pyplot(fig)

    # YEAR DISTRIBUTION
    st.write("## 7. Tahun Mobil")

    fig, ax = plt.subplots()
    sns.histplot(df['year'], bins=20, ax=ax)
    ax.set_title("Distribusi Tahun Mobil")
    st.pyplot(fig)

    # MILEAGE VS PRICE
    st.write("## 8. Mileage vs Price")

    fig, ax = plt.subplots()
    sns.scatterplot(x=df['mileage (km)'], y=df['price (Rp)'], ax=ax)
    ax.set_title("Mileage vs Price")
    st.pyplot(fig)

    # YEAR VS PRICE
    st.write("## 9. Year vs Price")

    fig, ax = plt.subplots()
    sns.scatterplot(x=df['year'], y=df['price (Rp)'], ax=ax)
    ax.set_title("Year vs Price")
    st.pyplot(fig)

# PREPROCESSING
elif menu == "Preprocessing":
    st.subheader("Preprocessing")

    df_clean = df.dropna().drop_duplicates()

    st.write(df_clean.head(800))
    st.write("Total Data (Row, Column):", df_clean.shape)

# TRAINING
elif menu == "Training":
    st.subheader("Model Training & Evaluation")

    st.markdown("### Model Used")
    st.info("""
    **Random Forest Regressor**

    This model works by:
    - Combining multiple decision trees
    - Reducing overfitting
    - Handling non-linear relationships effectively (e.g., car price patterns)
    """)

    st.markdown("### Training Pipeline")

    st.markdown("""
    #### 1. Data Preparation
    - Merged two datasets (used_car & used_car_data_new)
    - Data cleaning: removed missing values & duplicates
    - Text normalization (lowercase & trimming)

    #### 2. Feature Engineering
    - Created new feature:
      - `car_age = 2026 - year`

    #### 3. Encoding
    - Label Encoding → transmission
    - One-Hot Encoding → brand & model

    #### 4. Target Transformation
    - Applied `log1p(price)` to:
      - Reduce skewness
      - Improve model stability

    #### 5. Data Splitting
    - Training set: 80%
    - Testing set: 20%

    #### 6. Model Training
    - RandomForestRegressor (n_estimators=150)
    """)

    st.markdown("### Model Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric("MAE", "Rp 22.9M")
    col2.metric("RMSE", "Rp 35.8M")
    col3.metric("R² Score", "0.59")

    st.caption("Evaluation performed on test data after inverse log transformation")

    st.markdown("### Feature Importance")

    try:
        importances = model.feature_importances_
        feature_names = model_columns

        fi_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values(by="importance", ascending=False).head(10)

        fig, ax = plt.subplots()
        ax.barh(fi_df["feature"], fi_df["importance"])
        ax.invert_yaxis()

        st.pyplot(fig)

    except:
        st.warning("Feature importance is not available")

    st.markdown("### Conclusion")

    st.success("""
    - Mileage and car age are strong predictors of price
    - Brand and model significantly influence the prediction
    """)

# PREDICTION
elif menu == "Prediction":
    st.subheader("Input to predict used car price:")

    brand_input = st.text_input("Car Brand (Toyota, Honda, Daihatsu, ...)")
    model_input = st.text_input("Car Model (Inova, Brio, Agya, ...)")
    transmission = st.selectbox(
        "Transmission",
        options=le_trans.classes_
    )

    year = st.number_input("Year", 2000, 2026, 2020)
    mileage = st.number_input("Mileage (km)", 0, 500000, 50000)

    if st.button("Predict Price"):

        # BRAND MATCH
        brand_matches = smart_match(brand_input, valid_brands)

        if not brand_matches:
            st.error("Brand Not Found")
            st.stop()

        brand = brand_matches[0]

        # MODEL MATCH (FIXED)
        filtered_models = brand_model_map.get(brand, valid_models)

        model_matches = smart_match(model_input, filtered_models)

        if not model_matches:
            st.error("Model Not Found")
            st.stop()

        car_name = model_matches[0]

        # TRANSMISSION MATCH
        transmission = transmission

        st.info(f"""
        Interpretation results:
        - Brand: {brand}
        - Model: {car_name}
        - Transmission: {transmission}
        """)

        # FEATURE ENGINEERING
        car_age = 2026 - year

        input_df = pd.DataFrame(0, index=[0], columns=model_columns)

        input_df['mileage (km)'] = mileage
        input_df['car_age'] = car_age
        input_df['transmission'] = le_trans.transform([transmission])[0]

        # BRAND SET
        brand_col = f"brand_{brand}"

        if brand_col in model_columns:
            input_df[brand_col] = 1
        else:
            st.error("Brand not recognized by model")
            st.stop()

        # MODEL SET (FIX)
        model_col = f"model_{car_name}"

        if model_col in model_columns:
            input_df[model_col] = 1
        else:
            st.warning("The model doesn't exactly exist, looking for the closest one")

            model_candidates = [
                col.replace("model_", "")
                for col in model_columns
                if col.startswith("model_")
            ]

            closest = difflib.get_close_matches(car_name, model_candidates, n=1, cutoff=0.3)

            if closest:
                fixed_model = closest[0]
                input_df[f"model_{fixed_model}"] = 1
                st.success(f"Replaced to the nearest model: {fixed_model}")
            else:
                st.error("There are no similar models in training")
                st.stop()
        
        # PREDICT
        pred_log = model.predict(input_df)[0]
        pred_log = np.clip(pred_log, 0, 20)
        pred = np.expm1(pred_log)

        if np.isinf(pred) or np.isnan(pred):
            st.error("Prediction failed")
        else:
            st.success(f"Price Prediction: Rp {int(pred):,}")