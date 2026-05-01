# Used Car Price Prediction Indonesia

A machine learning project to predict the price of used cars in Indonesia using historical data and advanced predictive modeling techniques.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Workflow](#workflow)
- [Performance Metrics](#performance-metrics)
- [Technologies](#technologies)

## 🎯 Project Overview

This project aims to predict the price of used cars in Indonesia by analyzing various features such as:
- Brand and model of the car
- Year of manufacture
- Mileage (in kilometers)
- Transmission type (manual or automatic)

The prediction system uses a **Random Forest Regressor** trained on historical used car data and provides an interactive web interface built with Streamlit for easy access and predictions.

## ✨ Features

- **Smart Brand & Model Matching**: Intelligent fuzzy matching for user input with suggestions and fallback options
- **Interactive Web Interface**: User-friendly Streamlit-based application for making predictions
- **Data Preprocessing**: Comprehensive cleaning and normalization of data
- **Exploratory Data Analysis**: Statistical analysis and visualization of the dataset
- **Feature Engineering**: Advanced feature extraction and transformation
- **Model Training & Evaluation**: RandomForest model with performance metrics
- **Real-time Predictions**: Predict car prices based on user-provided information

## 📁 Project Structure

```
machine-learning-assignment/
├── app.py                          # Main Streamlit web application
├── trainmodel.py                   # Model training script
├── preprocessing.py                # Data preprocessing utilities
├── eda.py                          # Exploratory Data Analysis
├── prediksi.py                     # Prediction utilities with smart matching
├── requirements.txt                # Project dependencies
├── README.md                       # This file
├── dataset/
│   ├── used_car.csv               # First dataset of used cars
│   └── used_car_data_new.csv      # Second dataset of used cars
└── model/
    ├── model.pkl                  # Trained Random Forest model
    ├── le_trans.pkl               # Label encoder for transmission
    └── model_columns.pkl          # Feature columns used in training
```

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd machine-learning-assignment
   ```

2. **Create a virtual environment** (optional but recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### Train the Model

To retrain the model with the dataset:

```bash
python trainmodel.py
```

This will:
- Load and merge the two datasets
- Clean and preprocess the data
- Perform outlier removal
- Apply feature engineering
- Train the Random Forest model
- Save the trained model and encoders to the `model/` directory

### Run the Web Application

To start the Streamlit web interface:

```bash
streamlit run app.py
```

Then open your browser and navigate to `http://localhost:8501` to access the application.

In the web interface, you can:
1. Select a car brand
2. Choose a car model
3. Enter the year of manufacture
4. Input the mileage in kilometers
5. Select the transmission type
6. Get an instant price prediction

### Make Predictions Programmatically

Use the `prediksi.py` module:

```python
from prediksi import smart_match, predict_price

# Get matching brands and models
brands = smart_match("toyota", valid_brands)
models = smart_match("avanza", valid_models)

# Make a prediction
predicted_price = predict_price(brand, model, year, mileage, transmission)
print(f"Predicted Price: Rp {predicted_price:,.0f}")
```

## 📊 Dataset

The project uses two datasets of used cars in Indonesia:

| Dataset | File | Features |
|---------|------|----------|
| Dataset 1 | `used_car.csv` | Brand, Model, Year, Mileage (km), Transmission, Price (Rp) |
| Dataset 2 | `used_car_data_new.csv` | Brand, Model, Year, Mileage (km), ID Transmission, Price (Rp) |

**Preprocessing steps:**
- Merge both datasets
- Standardize transmission field (map ID to manual/automatic)
- Normalize mileage units
- Remove duplicates and missing values
- Lowercase all text fields

## 🤖 Model Architecture

**Model Type:** Random Forest Regressor

**Configuration:**
- Number of estimators: 150
- Random state: 42
- Target transformation: Log transformation (log1p)

**Features:**
- Brand (one-hot encoded, drop first)
- Model (one-hot encoded, drop first)
- Year
- Mileage (km)
- Transmission (label encoded: manual=0, automatic=1)

**Preprocessing:**
- Target variable is log-transformed to handle price distribution
- Features are standardized through encoding and normalization

## 🔄 Workflow

```
Raw Data
    ↓
Load & Merge Datasets
    ↓
Data Cleaning (remove nulls, duplicates, normalize text)
    ↓
Exploratory Data Analysis (EDA)
    ↓
Outlier Detection & Removal
    ↓
Feature Engineering
    ↓
Encoding (Label Encoding for transmission, One-Hot for categorical)
    ↓
Train-Test Split (80-20)
    ↓
Model Training (Random Forest)
    ↓
Model Evaluation
    ↓
Model Serialization (saved to model/)
    ↓
Web Application (Streamlit)
```

## 📈 Performance Metrics

The model is evaluated using the following metrics:

- **Mean Absolute Error (MAE)**: Average absolute difference between predicted and actual prices
- **Root Mean Squared Error (RMSE)**: Square root of average squared differences
- **R² Score**: Proportion of variance explained by the model

Run `trainmodel.py` to see the current model performance on the test set.

## 🛠 Technologies

| Technology | Version | Purpose |
|-----------|---------|---------|
| Python | 3.7+ | Programming language |
| Streamlit | 1.52.1 | Web application framework |
| Pandas | 2.3.3 | Data manipulation and analysis |
| NumPy | 2.3.5 | Numerical computing |
| Scikit-Learn | 1.8.0 | Machine learning library |
| Matplotlib | 3.10.8 | Data visualization |
| Seaborn | 0.13.2 | Statistical data visualization |
| Joblib | 1.5.3 | Model serialization |

## 📝 Notes

- The model uses log transformation on the target variable (price) to handle skewed distributions
- Smart matching function helps users find valid brands and models even with typos or partial matches
- The application caches data loading for better performance
- All numerical features are scaled appropriately during preprocessing

## 🤝 Contributing

Feel free to fork this project and submit pull requests for any improvements.

## 📄 License

This project is open source and available under the MIT License.