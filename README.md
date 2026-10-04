# Customer Churn Prediction System

An end-to-end machine learning project that predicts whether a customer
is likely to churn, with a polished Streamlit UI on top.

## Project overview

- **Synthetic dataset** (3,000 customers) with rule-based — not random —
  churn labels: risk rises with low satisfaction, low usage, short
  tenure, high charges, many support tickets, and Basic-tier plans.
- **Full pipeline**: cleaning, encoding, scaling, EDA, visualization,
  model training (Logistic Regression, Decision Tree, Random Forest),
  GridSearchCV tuning, evaluation, and model persistence.
- **Three ways to use it**: a Streamlit web app, a console app, and a
  standalone HTML demo that needs no Python at all.

## Project structure

```
Customer-Churn-Prediction/
├── README.md
├── requirements.txt
├── app.py                     # Streamlit UI — streamlit run app.py
├── churn_prediction.py        # Interactive console predictor
├── churn-dashboard.html       # Standalone client-side demo (no backend)
├── data/customer_churn_dataset.csv
├── models/churn_model.pkl
├── images/                    # All generated charts
└── src/
    ├── config.py               # Paths, constants, hyperparameters
    ├── utils.py                 # Logging, pickle I/O, input validation
    ├── dataset_generator.py     # Synthetic data generation
    ├── preprocessing.py         # Cleaning, encoding, scaling, split
    ├── eda.py                   # Exploratory data analysis
    ├── visualization.py         # All charts
    ├── model_training.py        # Training + GridSearchCV tuning
    ├── evaluation.py            # Metrics, comparison, best-model pick
    └── predictor.py             # Training pipeline + inference class
```

## How to run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit app (recommended)

```bash
streamlit run app.py
```

Opens a browser tab with three views:
- **Predict** — enter a customer's profile, get an instant prediction
- **Dataset Insights** — churn distribution, correlations, charges by status
- **Model Comparison** — Accuracy/Precision/Recall/F1/ROC-AUC across all
  three models, plus feature importance

The model trains automatically on first run (takes under a minute) and
is cached afterward.

### 3. Or run the console app

```bash
python churn_prediction.py
```

### 4. Or just open the static demo

`churn-dashboard.html` needs no Python — double-click it, or use VS
Code's Live Server extension. It has the trained Logistic Regression
model's logic embedded directly in JavaScript, so predictions run
entirely in the browser.

### 5. Retrain from scratch

```bash
python -m src.predictor
```

Regenerates the dataset, retrains all three models, re-tunes Random
Forest, regenerates every chart in `images/`, and overwrites
`models/churn_model.pkl`.

## Model performance

Best model: **Logistic Regression**, selected automatically by F1 score.

| Model               | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.668    | 0.660     | 0.667  | 0.663    | 0.732   |
| Random Forest        | 0.648    | 0.642     | 0.639  | 0.641    | 0.698   |
| Decision Tree        | 0.602    | 0.579     | 0.684  | 0.627    | 0.640   |

(Figures are from the shipped dataset; re-running the pipeline
regenerates fresh data and may shift these slightly.)

## Future improvements

- Swap the synthetic dataset for a real one once available
- Add gradient boosting models (XGBoost, LightGBM)
- Deploy the Streamlit app (e.g., Streamlit Community Cloud)
- Add authentication and a database for logging predictions over time

## License

For educational/internship project use.
