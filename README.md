# Customer Churn Prediction (Demo)

![CI](https://github.com/rudhrestam/customer-churn-prediction/actions/workflows/ci.yml/badge.svg) ![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A churn-modelling demo in the "customer 360" style: assemble features from transactions, engagement, and feedback sentiment, train a classifier, and turn the scores into a short daily-insights summary a business team can act on.

> ⚠️ **Synthetic data only.** The script generates its own fake customer base with a fixed seed. No real customer data — anyone's — is used, and the patterns it "learns" are illustrative, not findings.

> This is a representative demo of patterns used in production systems, built with original code. It contains no employer code, data, or internal details.

## The idea

Churn models fail in predictable ways: they train on transactions alone and miss the human signals. This demo assembles three feature families, the way a customer-360 platform does at a conceptual level:

| Signal family | Features in this demo |
|---|---|
| Transactions | visits in last 90 days, average spend, days since last visit |
| Engagement | email clicks (30d), app sessions (30d) |
| Feedback | average sentiment score (−1 to +1) |
| Tenure | months as a customer |

The model (a random forest) scores every customer daily; the script prints accuracy, a classification report, the feature-importance ranking, and a short insights summary: how many customers are high-risk today, and which signal combination matters most.

## Run it

```bash
pip install -r requirements.txt
python churn_demo.py                  # 3,000 synthetic customers
python churn_demo.py --customers 8000 # bigger base, same seed logic
```

## Tech stack

Python 3.10+ · scikit-learn (RandomForest) · NumPy

## What it demonstrates

- Multi-source feature assembly (transactions + engagement + feedback)
- Honest evaluation: held-out test split, accuracy + per-class report
- Feature-importance explainability for non-technical stakeholders
- Turning model output into a stakeholder-readable daily summary
- Reproducible synthetic-data demos (fixed seed) for portfolio/CI use
