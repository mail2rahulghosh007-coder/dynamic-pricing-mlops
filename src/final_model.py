import pandas as pd
import numpy as np
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import joblib

# Load feature-engineered data
df = pd.read_csv("data/features_ready.csv")
df['Date'] = pd.to_datetime(df['Date'])
df['StateHoliday'] = df['StateHoliday'].astype(str)
df = pd.get_dummies(df, columns=['StateHoliday', 'StoreType', 'Assortment', 'PromoInterval'], drop_first=True)
df = df.sort_values('Date').reset_index(drop=True)

drop_cols = ['Sales', 'Date', 'Customers']
feature_cols = [col for col in df.columns if col not in drop_cols]
X = df[feature_cols]
y = df['Sales']

# Time-based split: same 80-20 split as before (full data this time)
split_date = df['Date'].quantile(0.8)
train_mask = df['Date'] <= split_date
test_mask = df['Date'] > split_date

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print("Train size:", X_train.shape, "Test size:", X_test.shape)

# Load best hyperparameters found during tuning
with open("models/best_params.json", "r") as f:
    best_params = json.load(f)

print("\nUsing best parameters:", best_params)

# Train final model with best parameters, on FULL 80% training data
final_model = XGBRegressor(**best_params, random_state=42)
final_model.fit(X_train, y_train)

# Evaluate on test set
preds = final_model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))

print(f"\nFinal Model MAE: {mae:.2f}")
print(f"Final Model RMSE: {rmse:.2f}")

avg_sales = y_test.mean()
print(f"MAE as % of average sales: {(mae/avg_sales)*100:.2f}%")

# Save the trained model for later use (API/deployment)
joblib.dump(final_model, "models/final_xgb_model.pkl")
print("\nModel saved as models/final_xgb_model.pkl")

# Also save the feature column order (important for prediction later)
with open("models/feature_columns.json", "w") as f:
    json.dump(feature_cols, f)
print("Feature columns saved for future predictions")