import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
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

# Use only the most recent 40% of data for tuning (faster, still time-ordered)
cutoff = int(len(df) * 0.6)
X_tune = X.iloc[cutoff:]
y_tune = y.iloc[cutoff:]

# Define parameter search space
param_dist = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5]
}

tscv = TimeSeriesSplit(n_splits=3)

search = RandomizedSearchCV(
    estimator=XGBRegressor(random_state=42),
    param_distributions=param_dist,
    n_iter=20,                     # try 20 random combinations
    scoring='neg_mean_absolute_error',
    cv=tscv,
    verbose=2,
    random_state=42,
    n_jobs=-1                      # use all CPU cores
)

print("Starting hyperparameter search... this will take a few minutes")
search.fit(X_tune, y_tune)

print("\nBest parameters found:")
print(search.best_params_)
print(f"\nBest MAE during search: {-search.best_score_:.2f}")

# Save best params to a file so we can reuse them later
import json
with open("models/best_params.json", "w") as f:
    json.dump(search.best_params_, f)
print("\nSaved best_params.json in models/ folder")