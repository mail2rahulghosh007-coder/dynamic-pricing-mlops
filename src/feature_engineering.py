import pandas as pd

# Load cleaned data
df = pd.read_csv("data/cleaned_merged.csv")
df['Date'] = pd.to_datetime(df['Date'])

# Sort by Store and Date (very important for lag features to work correctly)
df = df.sort_values(['Store', 'Date'])

# 1. Date-based features
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['WeekOfYear'] = df['Date'].dt.isocalendar().week

# 2. Lag features (previous sales for the same store)
df['Sales_Lag_1'] = df.groupby('Store')['Sales'].shift(1)   # yesterday's sales
df['Sales_Lag_7'] = df.groupby('Store')['Sales'].shift(7)   # same day last week

# 3. Rolling average features (recent trend)
df['Sales_RollingMean_7'] = df.groupby('Store')['Sales'].transform(lambda x: x.shift(1).rolling(7).mean())
df['Sales_RollingMean_30'] = df.groupby('Store')['Sales'].transform(lambda x: x.shift(1).rolling(30).mean())

# Drop rows with NaN created by lag/rolling (first few rows per store won't have history)
df = df.dropna()

print("Shape after feature engineering:", df.shape)
print(df[['Store','Date','Sales','Sales_Lag_1','Sales_Lag_7','Sales_RollingMean_7']].head(10))

# Save for model training
df.to_csv("data/features_ready.csv", index=False)
print("\nSaved features_ready.csv successfully")