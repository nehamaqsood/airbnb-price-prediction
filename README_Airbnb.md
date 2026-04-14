# Airbnb Price Prediction

A data mining project that predicts Airbnb listing prices using regression and ensemble models, built to help hosts set competitive and fair prices based on property characteristics, location, and amenities.

**Duke Fuqua | DECISION 520Q | Team 45**
Chris Chen, Ziling Chen, Krishna Gupta, Sam MacArthur, Neha Maqsood

---

## Problem Statement

Many Airbnb listings are priced inconsistently with their true market value — overpriced properties see lower booking rates, while underpriced ones leave money on the table. This project uses data-driven modeling to identify what actually drives listing prices and flag properties that are significantly over or undervalued.

---

## Dataset

70,000+ Airbnb listings across major U.S. cities (New York City, San Francisco, Washington D.C., Los Angeles, and others) with 29 variables covering property details, host attributes, location, amenities, and customer engagement metrics.

Key preprocessing steps:
- Extracted 127 individual amenity boolean columns from unstructured text; retained 28 with 10,000+ occurrences
- Applied log transformation to price to handle right-skew from luxury listings
- Grouped rare categories in `property_type`, `room_type`, and `cancellation_policy`
- Encoded all categorical variables for modeling

---

## Models

| Model | Test R² | Test RMSE |
|-------|---------|-----------|
| Multiple Linear Regression | 0.596 | — |
| Lasso Regression | 0.5815 | 0.4653 |
| Linear Regression with Interactions | 0.6003 | — |
| **Random Forest (best)** | **0.6875** | **0.4007** |

Each model serves a different purpose. Linear and Lasso models are interpretable and useful for explaining pricing drivers to hosts. The interaction model captures joint feature effects. Random Forest achieves the highest accuracy and handles nonlinear relationships automatically.

---

## Key Findings

- **Room type, bedrooms, and accommodates** are the strongest price predictors across all models
- **Location** matters significantly — West Coast cities generally show lower prices after controlling for other variables
- **Amenities** like TV, elevator, and dryer rank in the top 20 feature importances
- Air conditioning is no longer a price premium in LA and DC — it's an expected baseline
- The Random Forest classified listings into: **undervalued (10%)**, **overvalued (10%)**, and **fairly priced (80%)**
  - Undervalued listings: avg. actual price $74 vs. predicted $141
  - Overvalued listings: avg. actual price $429 vs. predicted $184

---

## Business Application

Integrating the Random Forest model into Airbnb's host dashboard as a pricing recommendation tool could improve booking conversion rates, increase host earnings, and reduce customer churn from perceived unfair pricing. A 5–10% improvement in pricing precision at Airbnb's scale would translate to significant revenue impact.

---

## Files

- `airbnb_coding.py` — full data cleaning, modeling, and evaluation code
- `airbnb_original.csv` — raw dataset
- `Final_Report.pdf` — written report covering all six CRISP-DM phases
- `Decision_520.pptx` — presentation slides

---

## Running the Code

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python airbnb_coding.py
```
