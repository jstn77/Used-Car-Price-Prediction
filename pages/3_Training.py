import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(page_title="Model Training - Used Car Price Prediction", layout="wide")

st.title("🤖 Model Training & Evaluation")
st.markdown("""
This page explains the Machine Learning model development process used to predict used car prices.
""")

# --- SECTION 1: MODEL ARCHITECTURE ---
st.header("1. Model Architecture")
col1, col2 = st.columns([1, 2])

with col1:
    st.info("### Random Forest Regressor")
    st.write("""
    **Why choose this model?**
    - **Robust**: Handles outliers exceptionally well.
    - **Non-linear**: Captures complex price patterns effectively.
    - **Ensemble**: Combines multiple decision trees for high accuracy.
    """)

with col2:
    st.subheader("Training Pipeline")
    with st.expander("Click to see step-by-step details"):
        st.markdown("""
        1. **Data Cleaning**: Removed missing values and duplicate records.
        2. **Feature Engineering**: Created a new `car_age` feature (2026 - year).
        3. **Target Transformation**: Applied `log1p` to prices to stabilize variance.
        4. **Encoding**: Converted categorical text (Brand/Model) into numerical data.
        5. **Data Splitting**: Divided data into 80% Training and 20% Testing sets.
        """)

st.divider()

# --- SECTION 2: PERFORMANCE METRICS ---
st.header("2. Performance Metrics")
st.write("Model evaluation results on test data (after inverse log transformation):")

m1, m2, m3 = st.columns(3)
m1.metric("MAE (Mean Absolute Error)", "Rp 22.9M", help="Average difference between predicted and actual price")
m2.metric("RMSE (Root Mean Square Error)", "Rp 35.8M", help="Standard deviation of prediction errors")
m3.metric("R² Score", "0.59", help="How well the model explains price variance (Scale 0-1)")

st.divider()

# --- SECTION 3: FEATURE IMPORTANCE ---
st.header("3. Key Price Drivers")
st.write("Visualization of which variables most influence the price prediction:")

try:
    # Load model and columns from the model folder
    model = joblib.load("model/model.pkl")
    model_columns = joblib.load("model/model_columns.pkl")
    
    importances = model.feature_importances_
    fi_df = pd.DataFrame({
        "Feature": model_columns,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(fi_df["Feature"], fi_df["Importance"], color='teal')
    ax.invert_yaxis()
    plt.xlabel("Importance Level")
    st.pyplot(fig)
    
    st.success("""
    **Conclusion:**
    - **Mileage (KM)** and **Car Age** are the strongest factors in determining price.
    - Specific **Brands** and **Models** also provide significant influence.
    """)

except Exception as e:
    st.warning("Feature Importance chart could not be displayed. Ensure files in 'model/' folder are available.")
    st.error(f"Error: {e}")

st.divider()
st.markdown("### 💡 Note")
st.caption("This model is trained using the latest Indonesian used car market data up to 2026.")