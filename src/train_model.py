import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import numpy as np

# Load feature-engineered data
df = pd.read_csv("data/features_ready.csv")
df['Date'] = pd.to_datetime(df['Date'])

# Convert categorical columns to numeric (XGBoost needs numbers, not text)
df['StateHoliday'] = df['StateHoliday'].astype(str)
df = pd.get_dummies(df, columns=['StateHoliday', 'StoreType', 'Assortment', 'PromoInterval'], drop_first=True)

# Define which columns are features (inputs) vs target (what we predict)
drop_cols = ['Sales', 'Date', 'Customers']  # Customers dropped: not known in advance for future prediction
feature_cols = [col for col in df.columns if col not in drop_cols]

X = df[feature_cols]
y = df['Sales']

# Time-based split: train on older data, test on most recent data (NEVER shuffle time series data)
split_date = df['Date'].quantile(0.8)  # roughly 80% earliest dates for training
train_mask = df['Date'] <= split_date
test_mask = df['Date'] > split_date

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print("Train size:", X_train.shape, "Test size:", X_test.shape)

# Train XGBoost model
model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Predict on test set
preds = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# Naive baseline: predict today's sales = yesterday's sales (Sales_Lag_1)
naive_preds = X_test['Sales_Lag_1']

naive_mae = mean_absolute_error(y_test, naive_preds)
naive_rmse = np.sqrt(mean_squared_error(y_test, naive_preds))

print(f"\nNaive Baseline MAE: {naive_mae:.2f}")
print(f"Naive Baseline RMSE: {naive_rmse:.2f}")

print(f"\nImprovement over naive baseline: {((naive_mae - mae) / naive_mae * 100):.2f}%")

# Stronger baseline: predict today's sales = same day last week (Sales_Lag_7)
naive_week_preds = X_test['Sales_Lag_7']

naive_week_mae = mean_absolute_error(y_test, naive_week_preds)
naive_week_rmse = np.sqrt(mean_squared_error(y_test, naive_week_preds))

print(f"\nNaive Weekly Baseline (same day last week) MAE: {naive_week_mae:.2f}")
print(f"Naive Weekly Baseline RMSE: {naive_week_rmse:.2f}")

print(f"\nImprovement over WEEKLY naive baseline: {((naive_week_mae - mae) / naive_week_mae * 100):.2f}%")

# Also check: what % of average sales does our MAE represent?
avg_sales = y_test.mean()
print(f"\nAverage actual sales in test set: {avg_sales:.2f}")
print(f"Our model's MAE as % of average sales: {(mae/avg_sales)*100:.2f}%")