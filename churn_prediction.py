#!/usr/bin/env python3
"""Interactive console app: predicts churn for a customer you describe."""
from __future__ import annotations
import sys

from src.config import GENDERS, MODEL_PATH, PAYMENT_METHODS, SUBSCRIPTION_TYPES
from src.predictor import ChurnPredictor, InvalidCustomerInputError, train_and_save_pipeline
from src.utils import get_logger, prompt_choice, prompt_float, prompt_int

logger = get_logger(__name__)

BANNER = """
==============================================================
   CUSTOMER CHURN PREDICTION SYSTEM — Console Predictor
==============================================================
"""


def collect_input() -> dict:
    print("\nEnter the customer's details:\n")
    return {
        "Age": prompt_int("Age (18-100): ", 18, 100),
        "Gender": prompt_choice("Gender", GENDERS),
        "SubscriptionType": prompt_choice("Subscription Type", SUBSCRIPTION_TYPES),
        "MonthlyCharges": prompt_float("Monthly Charges ($): ", 0.0, 1000.0),
        "TenureMonths": prompt_int("Tenure (months): ", 0, 100),
        "SupportTickets": prompt_int("Support Tickets: ", 0, 50),
        "PaymentMethod": prompt_choice("Payment Method", PAYMENT_METHODS),
        "SatisfactionScore": prompt_int("Satisfaction Score (1-10): ", 1, 10),
        "UsageHoursPerWeek": prompt_float("Usage Hours per Week: ", 0.0, 168.0),
    }


def show_result(result) -> None:
    print("\n" + "=" * 55)
    icon = "⚠️ " if "Churn" in result.prediction else "✅ "
    print(f"{icon}{result.prediction}")
    print(f"   Churn Probability:  {result.churn_probability:.1%}")
    print(f"   Retain Probability: {result.retain_probability:.1%}")
    print(f"   Model Used:         {result.model_used}")
    print("=" * 55)


def ensure_model() -> None:
    if MODEL_PATH.exists():
        return
    print(f"No trained model found at '{MODEL_PATH}'.")
    if input("Train it now? [y/n]: ").strip().lower() != "y":
        print("Cannot proceed without a model. Exiting."); sys.exit(1)
    print("\nTraining... this takes a minute.\n")
    train_and_save_pipeline()
    print("\nDone.\n")


def main() -> None:
    print(BANNER)
    ensure_model()
    try:
        predictor = ChurnPredictor()
    except (FileNotFoundError, IOError) as e:
        print(f"Could not load model: {e}"); sys.exit(1)

    print(f"Model loaded: {predictor.model_name}\n")
    while True:
        try:
            result = predictor.predict(collect_input())
            show_result(result)
        except InvalidCustomerInputError as e:
            print(f"\nInvalid input: {e}\n"); continue
        if input("\nPredict another? [y/n]: ").strip().lower() != "y":
            print("\nGoodbye!"); break


if __name__ == "__main__":
    main()
