import pandas as pd

# Load the two files
train = pd.read_csv("data/train.csv")
store = pd.read_csv("data/store.csv")

# Convert Date column to proper datetime type
train['Date'] = pd.to_datetime(train['Date'])

# Merge train and store data using 'Store' column as the common key
merged = pd.merge(train, store, on='Store', how='left')

# Fill missing values in store-related columns
merged['CompetitionDistance'] = merged['CompetitionDistance'].fillna(merged['CompetitionDistance'].median())
merged['CompetitionOpenSinceMonth'] = merged['CompetitionOpenSinceMonth'].fillna(0)
merged['CompetitionOpenSinceYear'] = merged['CompetitionOpenSinceYear'].fillna(0)
merged['Promo2SinceWeek'] = merged['Promo2SinceWeek'].fillna(0)
merged['Promo2SinceYear'] = merged['Promo2SinceYear'].fillna(0)
merged['PromoInterval'] = merged['PromoInterval'].fillna('None')

# Keep only rows where store was open (closed store = 0 sales, not useful for price prediction)
merged = merged[merged['Open'] == 1]

# Check final shape and missing values
print("Final merged shape:", merged.shape)
print("\nMissing values per column:")
print(merged.isnull().sum())

# Save cleaned data for next steps
merged.to_csv("data/cleaned_merged.csv", index=False)
print("\nSaved cleaned_merged.csv successfully")