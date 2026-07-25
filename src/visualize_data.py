import pandas as pd
import matplotlib.pyplot as plt

# Load the cleaned data
df = pd.read_csv("data/cleaned_merged.csv")
df['Date'] = pd.to_datetime(df['Date'])

# 1. Overall sales trend over time (aggregate all stores by date)
daily_sales = df.groupby('Date')['Sales'].sum()

plt.figure(figsize=(14, 5))
plt.plot(daily_sales)
plt.title("Total Sales Over Time (All Stores)")
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.savefig("notebooks/sales_trend.png")
plt.close()

# 2. Average sales by day of week
avg_by_dow = df.groupby('DayOfWeek')['Sales'].mean()

plt.figure(figsize=(8, 5))
avg_by_dow.plot(kind='bar')
plt.title("Average Sales by Day of Week (1=Monday, 7=Sunday)")
plt.xlabel("Day of Week")
plt.ylabel("Average Sales")
plt.tight_layout()
plt.savefig("notebooks/sales_by_dayofweek.png")
plt.close()

# 3. Effect of Promo on sales
avg_by_promo = df.groupby('Promo')['Sales'].mean()
print("Average sales with vs without promo:")
print(avg_by_promo)

# 4. Effect of StoreType on sales
avg_by_storetype = df.groupby('StoreType')['Sales'].mean()
print("\nAverage sales by store type:")
print(avg_by_storetype)

print("\nPlots saved in notebooks/ folder")