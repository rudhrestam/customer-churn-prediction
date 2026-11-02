#!/usr/bin/env python3
"""
Customer churn prediction (demo) — synthetic data only.

Builds a small "customer 360" style feature set from three signal types:
  * transactions  — visit frequency, average spend, days since last visit
  * engagement   — email / app click activity
  * feedback     — average sentiment score from customer feedback

Generates a synthetic customer base, labels churn with a noisy rule based
on the same signals a real model would learn from, trains a classifier,
reports accuracy, and prints a "daily insights" style summary — the kind
of output a stakeholder would read over coffee.

No real customer data is used anywhere in this demo.

Usage:
    python churn_demo.py [--customers 3000] [--seed 42]
"""

import argparse

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

FEATURES = [
    "visits_last_90d",
    "avg_spend",
    "days_since_last_visit",
    "email_clicks_30d",
    "app_sessions_30d",
    "feedback_sentiment",  # -1 (negative) .. +1 (positive)
    "tenure_months",
]


def generate_customers(n: int, seed: int):
    rng = np.random.default_rng(seed)
    visits = rng.poisson(6, n).clip(0, 40)
    avg_spend = rng.normal(24, 9, n).clip(3, 90)
    days_since = rng.gamma(2.2, 14, n).clip(0, 180)
    email_clicks = rng.poisson(3, n).clip(0, 30)
    app_sessions = rng.poisson(5, n).clip(0, 45)
    sentiment = rng.normal(0.15, 0.5, n).clip(-1, 1)
    tenure = rng.integers(1, 72, n)

    # Latent churn risk: disengagement + recency + negative feedback.
    risk = (
        -0.16 * visits
        + 0.035 * days_since
        - 0.14 * email_clicks
        - 0.10 * app_sessions
        - 1.15 * sentiment
        - 0.012 * tenure
        + rng.normal(0, 0.9, n)
    )
    churn = (risk > np.quantile(risk, 0.78)).astype(int)  # ~22% churn rate

    X = np.column_stack(
        [visits, avg_spend, days_since, email_clicks, app_sessions, sentiment, tenure]
    )
    return X, churn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--customers", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    X, y = generate_customers(args.customers, args.seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=args.seed, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, random_state=args.seed, n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)

    print(f"Synthetic customers: {args.customers}  ·  churn rate: {y.mean():.1%}")
    print(f"Test accuracy: {acc:.1%}\n")
    print(classification_report(y_test, pred, target_names=["stays", "churns"]))

    importance = sorted(zip(FEATURES, model.feature_importances_), key=lambda p: -p[1])
    print("What drives the predictions:")
    for name, weight in importance:
        print(f"  {name:<24} {'#' * int(weight * 40)} {weight:.2f}")

    # Daily-insights style summary over the full base.
    proba = model.predict_proba(X)[:, 1]
    at_risk = int((proba > 0.7).sum())
    print("\n--- Daily insights (demo) ---")
    print(f"Customers scored today:        {len(y)}")
    print(f"High-risk customers (p > 0.7): {at_risk}")
    print(f"Average predicted churn risk:  {proba.mean():.1%}")
    print("Suggested focus: recent visitors gone quiet + negative feedback "
          "is the strongest combined signal in this model.")


if __name__ == "__main__":
    main()
