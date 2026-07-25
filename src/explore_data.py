import pandas as pd

# Load the two files
train = pd.read_csv("data/train.csv")
store = pd.read_csv("data/store.csv")

# Basic look at train.csv
print("----- TRAIN DATA -----")
print(train.shape)          # (rows, columns)
print(train.columns.tolist())
print(train.head())
print(train.info())

# Basic look at store.csv
print("\n----- STORE DATA -----")
print(store.shape)
print(store.columns.tolist())
print(store.head())
print(store.info())

# Check: when store is closed, is sales always 0?
closed_stores = train[train['Open'] == 0]
print("\n----- CHECKING CLOSED STORES -----")
print("Number of rows where store is closed:", len(closed_stores))
print("Sales values when closed (should all be 0):")
print(closed_stores['Sales'].unique())