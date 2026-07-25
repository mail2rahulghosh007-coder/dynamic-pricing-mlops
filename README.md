# 💰 Dynamic Pricing Predictor

An end-to-end MLOps project that forecasts retail demand and recommends dynamic pricing using machine learning, built on the Rossmann Store Sales dataset (1M+ records, 1,115 stores).

🔗 **Live Demo:** [dynamic-pricing-predictor-rahul.streamlit.app](https://dynamic-pricing-predictor-rahul.streamlit.app)

---

## 📌 Problem Statement

Retailers need to price products dynamically based on demand signals (seasonality, promotions, competition) instead of using static pricing. This project predicts daily store-level demand and translates it into a data-driven price recommendation.

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Modeling | XGBoost, Scikit-learn |
| Experiment Tracking | MLflow |
| Data Versioning & Pipeline | DVC (`dvc.yaml` reproducible pipeline) |
| API | FastAPI |
| UI | Streamlit |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Deployment | Streamlit Community Cloud |

## 🔄 Pipeline Overview

```
Raw Data (train.csv + store.csv)
        ↓ [DVC tracked]
Data Cleaning & Merging
        ↓
Feature Engineering (lag features, rolling averages, date features)
        ↓
XGBoost Model Training + Hyperparameter Tuning (RandomizedSearchCV)
        ↓ [MLflow tracked]
Time-Series Cross-Validation & Evaluation
        ↓
Price Suggestion Logic (percentile-based demand thresholds)
        ↓
FastAPI Backend + Streamlit Frontend
        ↓ [Dockerized]
Deployed via CI/CD (GitHub Actions)
```

## 📊 Model Performance

- **MAE:** ₹609.17 (≈9% of average daily sales)
- **71.23% improvement** over a weekly-seasonality-aware naive baseline
- Validated using **5-fold Time Series Cross-Validation** (not random split, to avoid data leakage)
- Top predictive features: 30-day rolling average sales, Promo status, previous day's sales

## 🎯 Key Engineering Decisions

- Used **time-based train/test splits** throughout (never random) to reflect real-world forecasting conditions
- Compared model performance against **two baselines** (naive lag-1 and weekly-seasonal lag-7) to validate genuine model skill, not just pattern-copying
- Feature engineering avoided **data leakage** by shifting rolling averages by one day before computing
- Pricing thresholds were derived from **actual sales percentiles**, not arbitrary cutoffs

## 🚀 Running Locally

```bash
git clone https://github.com/mail2rahulghosh007-coder/dynamic-pricing-mlops.git
cd dynamic-pricing-mlops
pip install -r requirements.txt
dvc repro                 # reproduce the full data-to-model pipeline
streamlit run src/streamlit_app.py
```

Or with Docker:
```bash
docker build -t dynamic-pricing-app .
docker run -p 7860:7860 dynamic-pricing-app
```

## 📁 Project Structure

```
├── src/
│   ├── clean_data.py
│   ├── feature_engineering.py
│   ├── final_model.py
│   ├── price_logic.py
│   ├── api.py
│   └── streamlit_app.py
├── models/
├── data/
├── dvc.yaml
├── Dockerfile
├── .github/workflows/ci.yml
└── requirements.txt
```

## 🔮 Future Improvements

- Add data/concept drift monitoring (e.g., Evidently AI) for production retraining triggers
- Migrate DVC remote storage from local to cloud (S3/GCS) for full team collaboration
- Extend pricing logic from rule-based thresholds to a learned optimization model

---

**Author:** Rahul Ghosh
