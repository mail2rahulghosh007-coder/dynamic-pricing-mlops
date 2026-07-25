import pandas as pd

df = pd.read_csv("data/features_ready.csv")
print(df['Sales'].describe())
print("\nPercentiles:")
print("25th percentile:", df['Sales'].quantile(0.25))
print("50th percentile (median):", df['Sales'].quantile(0.50))
print("75th percentile:", df['Sales'].quantile(0.75))
print("90th percentile:", df['Sales'].quantile(0.90))