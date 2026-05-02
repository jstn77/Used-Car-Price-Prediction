import streamlit as st

st.set_page_config(page_title="Used Car Price Prediction", layout="wide", page_icon="🚗")

st.title("🚘Used Car Price Prediction")
st.markdown("""
**The Used Car Price Prediction application contains comprehensive data regarding the Indonesian automotive market, featuring various car brands, models, and specifications. This data is meticulously structured and analyzed to estimate the market value of a vehicle based on multiple parameters such as mileage, vehicle age, and transmission type.  
Have you ever thought about selling your car or buying a used one but felt unsure about the fair market price? FRET NOT!
            Using advanced Machine Learning techniques—specifically the Random Forest Regressor—this application provides you with data-driven answers. By simply filling in some general information about the vehicle, such as its brand, model, and year, our state-of-the-art service can predict the estimated price in seconds. It is your reliable companion for making informed decisions in the used car market!**
#### Use the sidebar menu to access:
1. **📂Dataset**: Looking at the raw data used.
2. **📊EDA**: Statistical analysis and data visualization.
3. **🤖Training**: Details of Machine Learning model performance.
4. **🚗Prediction**: A tool to predict your car's price.
""")