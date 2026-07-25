import streamlit as st
from price_logic import suggest_price

st.set_page_config(page_title="Dynamic Pricing Predictor", page_icon="💰")

st.title("💰 Dynamic Pricing Predictor")
st.write("Enter store and date details to get a demand forecast and suggested price.")

# Input fields
col1, col2 = st.columns(2)

with col1:
    store = st.number_input("Store ID", min_value=1, max_value=1115, value=1)
    day_of_week = st.selectbox("Day of Week (1=Mon, 7=Sun)", options=[1,2,3,4,5,6,7], index=4)
    promo = st.selectbox("Promo Running?", options=[0,1], index=1)
    school_holiday = st.selectbox("School Holiday?", options=[0,1], index=0)
    year = st.number_input("Year", value=2015)
    month = st.number_input("Month", min_value=1, max_value=12, value=12)

with col2:
    day = st.number_input("Day", min_value=1, max_value=31, value=20)
    week_of_year = st.number_input("Week of Year", min_value=1, max_value=53, value=51)
    sales_lag_1 = st.number_input("Yesterday's Sales", value=6000.0)
    sales_lag_7 = st.number_input("Same Day Last Week Sales", value=5800.0)
    rolling_7 = st.number_input("7-Day Avg Sales", value=5900.0)
    rolling_30 = st.number_input("30-Day Avg Sales", value=5700.0)

base_price = st.number_input("Base Price (₹)", value=100.0)

if st.button("Predict Price"):
    payload = {
        "Store": store,
        "DayOfWeek": day_of_week,
        "Promo": promo,
        "SchoolHoliday": school_holiday,
        "Year": year,
        "Month": month,
        "Day": day,
        "WeekOfYear": week_of_year,
        "Sales_Lag_1": sales_lag_1,
        "Sales_Lag_7": sales_lag_7,
        "Sales_RollingMean_7": rolling_7,
        "Sales_RollingMean_30": rolling_30,
        "CompetitionDistance": 1270.0,
        "CompetitionOpenSinceMonth": 9,
        "CompetitionOpenSinceYear": 2008,
        "Promo2": 0,
        "Promo2SinceWeek": 0,
        "Promo2SinceYear": 0
    }

    try:
        result = suggest_price(payload, base_price=base_price)

        st.success("Prediction complete!")
        st.metric("Predicted Sales", f"₹{result['predicted_sales']}")
        st.metric("Demand Level", result['demand_level'])
        st.metric("Suggested Price", f"₹{result['suggested_price']}")
    except Exception as e:
        st.error(f"Error: {e}")