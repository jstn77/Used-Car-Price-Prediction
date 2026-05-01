import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda(df):
    print(" EDA ")

    print("\nDeskripsi Harga:")
    print(df['price (Rp)'].describe())

    print("\nTop Brand:")
    print(df['brand'].value_counts().head())

    # Histogram harga
    plt.figure()
    sns.histplot(df['price (Rp)'], bins=30)
    plt.title("Distribusi Harga")
    plt.show()

    # Boxplot (cek outlier)
    plt.figure()
    sns.boxplot(x=df['price (Rp)'])
    plt.title("Boxplot Harga")
    plt.show()