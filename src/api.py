from fastapi import FastAPI
from pydantic import BaseModel
from src.price_logic import suggest_price

app = FastAPI(title="Dynamic Pricing API", description="Predicts demand and suggests price based on store/date features")

# Define the expected input structure (FastAPI validates this automatically)
class PricingRequest(BaseModel):
    Store: int
    DayOfWeek: int
    Promo: int
    SchoolHoliday: int
    Year: int
    Month: int
    Day: int
    WeekOfYear: int
    Sales_Lag_1: float
    Sales_Lag_7: float
    Sales_RollingMean_7: float
    Sales_RollingMean_30: float
    CompetitionDistance: float
    CompetitionOpenSinceMonth: float
    CompetitionOpenSinceYear: float
    Promo2: int
    Promo2SinceWeek: float
    Promo2SinceYear: float
    base_price: float = 100.0   # optional, defaults to 100 if not given


@app.get("/")
def home():
    return {"message": "Dynamic Pricing API is running. Go to /docs to try it out."}


@app.post("/predict")
def predict_price(request: PricingRequest):
    # Convert request into a plain dictionary
    input_dict = request.dict()
    base_price = input_dict.pop("base_price")   # remove base_price from features, use separately

    result = suggest_price(input_dict, base_price=base_price)
    return result