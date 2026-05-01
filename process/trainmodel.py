import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from process.preprocessing import load_and_merge, clean_data, remove_outlier, feature_engineering
from process.eda import run_eda

# LOAD & PREPROCESS
df = load_and_merge()
df = clean_data(df)

# EDA
run_eda(df)

# OUTLIER & FEATURE
df = remove_outlier(df)
df = feature_engineering(df)

# ENCODING
le_trans = LabelEncoder()
df['transmission'] = le_trans.fit_transform(df['transmission'])

df = pd.get_dummies(df, columns=['brand', 'model'], drop_first=True)

# SPLIT
X = df.drop(columns=['price (Rp)', 'year'])
y = np.log1p(df['price (Rp)'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TRAIN
model = RandomForestRegressor(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# EVALUATION
pred_log = model.predict(X_test)

pred = np.expm1(pred_log)
y_test = np.expm1(y_test)

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print("\n=== HASIL MODEL ===")
print(f"MAE  : Rp {mae:,.0f}")
print(f"RMSE : Rp {rmse:,.0f}")
print(f"R2   : {r2:.4f}")

# SAVE
joblib.dump(model, "model/model.pkl")
joblib.dump(le_trans, "model/le_trans.pkl")
joblib.dump(list(X.columns), "model/model_columns.pkl")

print("\nModel Saved!")