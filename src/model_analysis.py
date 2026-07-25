import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
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

# ---------- 1. TIME SERIES CROSS-VALIDATION ----------
tscv = TimeSeriesSplit(n_splits=5)
fold_scores = []

print("----- Time Series Cross-Validation (5 folds) -----")
for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    fold_scores.append(mae)
    print(f"Fold {fold+1}: MAE = {mae:.2f}")

print(f"\nAverage MAE across folds: {np.mean(fold_scores):.2f}")
print(f"Std deviation across folds: {np.std(fold_scores):.2f}")

# ---------- 2. FEATURE IMPORTANCE (using last fold's model) ----------
importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\n----- Top 10 Most Important Features -----")
print(importance.head(10))

plt.figure(figsize=(10, 6))
importance.head(10).plot(kind='barh')
plt.title("Top 10 Feature Importance (XGBoost)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("notebooks/feature_importance.png")
plt.close()

# ---------- 3. RESIDUAL ANALYSIS ----------
results_df = X_test.copy()
results_df['Actual'] = y_test.values
results_df['Predicted'] = preds
results_df['Error'] = results_df['Actual'] - results_df['Predicted']
results_df['AbsError'] = results_df['Error'].abs()

# Where is the model making the biggest mistakes?
print("\n----- Average Absolute Error by StoreType columns -----")
storetype_cols = [c for c in results_df.columns if c.startswith('StoreType_')]
for col in storetype_cols:
    subset = results_df[results_df[col] == 1]
    if len(subset) > 0:
        print(f"{col}: Avg Error = {subset['AbsError'].mean():.2f}, Count = {len(subset)}")

print("\nSaved feature_importance.png in notebooks/")