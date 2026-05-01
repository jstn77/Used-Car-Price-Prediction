import pandas as pd

def load_and_merge():
    df1 = pd.read_csv("dataset/used_car.csv")
    df2 = pd.read_csv("dataset/used_car_data_new.csv")

    # mapping dataset 2
    df2['transmission'] = df2['id_transmission'].map({1: 'manual', 2: 'automatic'})
    df1['mileage (km)'] = df1['mileage (km)'] * 1000

    df = pd.concat([df1, df2], ignore_index=True)

    df = df[['brand', 'model', 'year', 'mileage (km)', 'transmission', 'price (Rp)']]

    return df


def clean_data(df):
    df['brand'] = df['brand'].astype(str).str.lower().str.strip()
    df['model'] = df['model'].astype(str).str.lower().str.strip()
    df['transmission'] = df['transmission'].astype(str).str.lower().str.strip()

    df = df.drop_duplicates()
    df = df.dropna()

    return df


def remove_outlier(df):
    Q1 = df['price (Rp)'].quantile(0.25)
    Q3 = df['price (Rp)'].quantile(0.75)
    IQR = Q3 - Q1

    df = df[
        (df['price (Rp)'] >= Q1 - 1.5 * IQR) &
        (df['price (Rp)'] <= Q3 + 1.5 * IQR)
    ]

    return df


def feature_engineering(df):
    df['car_age'] = 2026 - df['year']
    return df