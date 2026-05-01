import numpy as np
import pandas as pd
import joblib
import difflib

# LOAD MODEL & FILE
model = joblib.load("model/model.pkl")
le_trans = joblib.load("model/le_trans.pkl")
model_columns = joblib.load("model/model_columns.pkl")

# LOAD DATA
df1 = pd.read_csv("dataset/used_car.csv", usecols=['brand', 'model'])
df2 = pd.read_csv("dataset/used_car_data_new.csv", usecols=['brand', 'model'])

df_temp = pd.concat([df1, df2])

valid_brands = df_temp['brand'].astype(str).str.lower().str.strip().dropna().unique()
valid_models = df_temp['model'].astype(str).str.lower().str.strip().dropna().unique()

# SMART MATCH FUNCTION
def smart_match(input_str, choices):
    input_str = input_str.lower().strip()

    # exact match
    if input_str in choices:
        return [input_str]

    # contains match
    contains_matches = [c for c in choices if input_str in c]
    if contains_matches:
        return contains_matches[:5]

    # fuzzy match
    fuzzy_matches = difflib.get_close_matches(input_str, choices, n=5, cutoff=0.3)
    return fuzzy_matches


print("=== Prediksi Harga Mobil ===")

# INPUT USER
brand_input = input("Masukkan brand: ").lower().strip()
model_input = input("Masukkan model: ").lower().strip()
trans_input = input("Masukkan transmission (manual/automatic): ").lower().strip()

year = int(input("Masukkan tahun: "))
mileage = int(input("Masukkan mileage (km): "))

# MATCH BRAND
brand_matches = smart_match(brand_input, valid_brands)

if not brand_matches:
    print("Brand tidak ditemukan")
    print("Contoh:", valid_brands[:5])
    exit()

if len(brand_matches) == 1:
    brand = brand_matches[0]
else:
    print("\nMaksud brand kamu:")
    for i, b in enumerate(brand_matches):
        print(f"{i}. {b}")
    brand = brand_matches[int(input("Pilih nomor: "))]

# MATCH MODEL
filtered_models = [m for m in valid_models if brand in m]

if not filtered_models:
    filtered_models = valid_models

model_matches = smart_match(model_input, filtered_models)

if not model_matches:
    print("Model tidak ditemukan untuk brand tersebut")
    exit()

if len(model_matches) == 1:
    car_name = model_matches[0]
else:
    print("\nMaksud model kamu:")
    for i, m in enumerate(model_matches):
        print(f"{i}. {m}")
    car_name = model_matches[int(input("Pilih nomor model: "))]

# MATCH TRANSMISSION
trans_matches = smart_match(trans_input, le_trans.classes_)

if not trans_matches:
    print("Transmission tidak ditemukan")
    exit()

transmission = trans_matches[0]

# PRINT INTERPRETASI
print("\nHasil interpretasi input:")
print("Brand       :", brand)
print("Model       :", car_name)
print("Transmission:", transmission)

# FEATURE ENGINEERING
car_age = 2026 - year

input_df = pd.DataFrame(0, index=[0], columns=model_columns)

input_df['mileage (km)'] = mileage
input_df['car_age'] = car_age
input_df['transmission'] = le_trans.transform([transmission])[0]

# ONE HOT MATCH
brand_col = f"brand_{brand}"
model_col = f"model_{car_name}"

if brand_col in model_columns:
    input_df[brand_col] = 1
else:
    print("Brand tidak ada di training model")

if model_col in model_columns:
    input_df[model_col] = 1
else:
    print("Model tidak ada di training model")

# PREDIKSI
pred_log = model.predict(input_df)[0]

# FIX OVERFLOW
pred_log = np.clip(pred_log, 0, 20)

pred = np.expm1(pred_log)

# VALIDASI OUTPUT
if np.isinf(pred) or np.isnan(pred):
    print("Prediksi gagal (input tidak dikenali model)")
else:
    print("\nEstimasi Harga Mobil:")
    print(f"Rp {int(pred):,}")