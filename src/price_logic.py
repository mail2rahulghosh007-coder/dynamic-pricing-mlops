import joblib
import json
import pandas as pd

# Load the trained model and feature columns (saved earlier)
model = joblib.load("models/final_xgb_model.pkl")

with open("models/feature_columns.json", "r") as f:
    feature_cols = json.load(f)

def suggest_price(input_features: dict, base_price: float = 100.0):
    """
    Takes store/date related features, predicts demand (sales),
    and suggests a price adjustment based on predicted demand level.
    """
    # Build a single-row DataFrame matching the model's expected columns
    input_df = pd.DataFrame([input_features])

    # Make sure all expected columns exist (fill missing ones with 0)
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_cols]  # correct column order

    # Predict demand (sales)
    predicted_sales = model.predict(input_df)[0]

    # Data-driven pricing logic (based on actual percentiles of Sales distribution)
    if predicted_sales > 10078:      # above 90th percentile
        price_multiplier = 1.15
        demand_level = "Very High"
    elif predicted_sales > 7938:     # above 75th percentile
        price_multiplier = 1.08
        demand_level = "High"
    elif predicted_sales > 4687:     # above 25th percentile (normal range)
        price_multiplier = 1.0
        demand_level = "Normal"
    else:                             # below 25th percentile
        price_multiplier = 0.92
        demand_level = "Low"

    suggested_price = round(base_price * price_multiplier, 2)

    return {
        "predicted_sales": round(float(predicted_sales), 2),
        "demand_level": demand_level,
        "suggested_price": suggested_price
    }


# Quick test (only runs when this file is executed directly)
# Quick test with multiple scenarios
if __name__ == "__main__":

    test_cases = {
        "High demand (Friday, Promo ON, high lag sales)": {
            "Store": 1, "DayOfWeek": 5, "Promo": 1, "SchoolHoliday": 0,
            "Year": 2015, "Month": 12, "Day": 20, "WeekOfYear": 51,
            "Sales_Lag_1": 11000, "Sales_Lag_7": 10500,
            "Sales_RollingMean_7": 10800, "Sales_RollingMean_30": 10200,
            "CompetitionDistance": 1270.0, "CompetitionOpenSinceMonth": 9,
            "CompetitionOpenSinceYear": 2008, "Promo2": 0,
            "Promo2SinceWeek": 0, "Promo2SinceYear": 0
        },
        "Low demand (weekday, no promo, low lag sales)": {
            "Store": 1, "DayOfWeek": 2, "Promo": 0, "SchoolHoliday": 0,
            "Year": 2015, "Month": 2, "Day": 10, "WeekOfYear": 7,
            "Sales_Lag_1": 3000, "Sales_Lag_7": 2800,
            "Sales_RollingMean_7": 2900, "Sales_RollingMean_30": 3100,
            "CompetitionDistance": 1270.0, "CompetitionOpenSinceMonth": 9,
            "CompetitionOpenSinceYear": 2008, "Promo2": 0,
            "Promo2SinceWeek": 0, "Promo2SinceYear": 0
        },
        "Very high demand (holiday season, promo ON)": {
            "Store": 1, "DayOfWeek": 6, "Promo": 1, "SchoolHoliday": 1,
            "Year": 2015, "Month": 12, "Day": 24, "WeekOfYear": 52,
            "Sales_Lag_1": 14000, "Sales_Lag_7": 13500,
            "Sales_RollingMean_7": 13800, "Sales_RollingMean_30": 12500,
            "CompetitionDistance": 1270.0, "CompetitionOpenSinceMonth": 9,
            "CompetitionOpenSinceYear": 2008, "Promo2": 0,
            "Promo2SinceWeek": 0, "Promo2SinceYear": 0
        },
        "Normal weekday, moderate lag sales": {
            "Store": 1, "DayOfWeek": 3, "Promo": 0, "SchoolHoliday": 0,
            "Year": 2015, "Month": 6, "Day": 15, "WeekOfYear": 24,
            "Sales_Lag_1": 6200, "Sales_Lag_7": 6000,
            "Sales_RollingMean_7": 6100, "Sales_RollingMean_30": 6000,
            "CompetitionDistance": 1270.0, "CompetitionOpenSinceMonth": 9,
            "CompetitionOpenSinceYear": 2008, "Promo2": 0,
            "Promo2SinceWeek": 0, "Promo2SinceYear": 0
        }
    }

    for label, features in test_cases.items():
        result = suggest_price(features, base_price=100.0)
        print(f"\n{label}")
        print(result)