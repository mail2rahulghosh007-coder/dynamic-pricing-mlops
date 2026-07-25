import pandas as pd
import numpy as np
import json
import mlflow
import mlflow.xgboost
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

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

split_date = df['Date'].quantile(0.8)
train_mask = df['Date'] <= split_date
test_mask = df['Date'] > split_date

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

with open("models/best_params.json", "r") as f:
    best_params = json.load(f)

# Set experiment name (creates a folder to group all related runs)
mlflow.set_experiment("dynamic-pricing-forecast")

with mlflow.start_run(run_name="xgboost_tuned_final"):
    # Log the hyperparameters we used
    mlflow.log_params(best_params)

    # Train model
    model = XGBRegressor(**best_params, random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae_pct = (mae / y_test.mean()) * 100

    # Log metrics
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("MAE_pct_of_avg_sales", mae_pct)

    # Log the model itself (saved inside MLflow's tracking system)
    mlflow.xgboost.log_model(model, "model")

    print(f"Logged run to MLflow. MAE: {mae:.2f}, RMSE: {rmse:.2f}")

print("\nRun 'mlflow ui' in terminal to view results in browser")