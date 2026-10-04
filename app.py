"""
Streamlit UI for the Customer Churn Prediction System.

Run with:
    streamlit run app.py
"""
from __future__ import annotations
import pandas as pd
import streamlit as st

from src.config import (
    DATASET_PATH, GENDERS, IMAGES_DIR, MODEL_PATH, PAYMENT_METHODS, SUBSCRIPTION_TYPES,
)
from src.predictor import ChurnPredictor, InvalidCustomerInputError, train_and_save_pipeline

st.set_page_config(page_title="Churn Signal", page_icon="📡", layout="wide")

st.markdown("""
<style>
:root { --teal:#0F5257; --coral:#C8432F; --ink:#16231F; --sub:#54635D; }

/* Force a consistent light surface + explicit text color everywhere,
   so the page never relies on inheriting a browser/OS dark-mode color
   that would make text invisible against our light background. */
.stApp { background:#F3F5F2; color:var(--ink); }
.stApp, .stApp p, .stApp span, .stApp label, .stApp li,
.stApp h1, .stApp h2, .stApp h3, .stApp h4,
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stCaptionContainer"],
[data-testid="stWidgetLabel"] p,
[data-testid="stMetricLabel"],
[data-testid="stTabs"] button p {
    color: var(--ink) !important;
}
[data-testid="stCaptionContainer"] { color: var(--sub) !important; }
[data-testid="stHeader"] { background: transparent; }

h1, h2, h3 { font-family:'Georgia', serif; }

.metric-card {
    background:#FFFFFF; border:1px solid #DCE3DD; border-radius:14px;
    padding:18px 20px; text-align:center;
}
.metric-card .k { font-size:26px; font-weight:700; color:var(--ink); }
.metric-card .l { font-size:12.5px; color:var(--sub); margin-top:2px; }
.badge { padding:8px 18px; border-radius:999px; font-weight:600; display:inline-block; }
.badge.churn { background:#FBE8E3; color:#C8432F !important; }
.badge.stay { background:#E4F0EE; color:#0F5257 !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_predictor():
    if not MODEL_PATH.exists():
        train_and_save_pipeline()
    return ChurnPredictor()


@st.cache_data
def get_dataset():
    return pd.read_csv(DATASET_PATH) if DATASET_PATH.exists() else None


st.title("📡 Churn Signal")
st.caption("Customer churn prediction — enter a profile and get an instant, explained prediction.")

tab_predict, tab_insights, tab_models = st.tabs(["🔮 Predict", "📊 Dataset Insights", "🏆 Model Comparison"])

# ---------------------------------------------------------------
with tab_predict:
    predictor = get_predictor()
    st.subheader("Customer profile")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Age", 18, 75, 39)
        gender = st.selectbox("Gender", GENDERS)
        subscription = st.selectbox("Subscription Type", SUBSCRIPTION_TYPES)
    with col2:
        charges = st.slider("Monthly Charges ($)", 9.99, 119.99, 40.0)
        tenure = st.slider("Tenure (months)", 0, 72, 17)
        tickets = st.slider("Support Tickets", 0, 12, 2)
    with col3:
        payment = st.selectbox("Payment Method", PAYMENT_METHODS)
        satisfaction = st.slider("Satisfaction Score (1-10)", 1, 10, 6)
        usage = st.slider("Usage Hours/Week", 0.0, 40.0, 12.0)

    if st.button("Predict", type="primary"):
        record = {
            "Age": age, "Gender": gender, "SubscriptionType": subscription,
            "MonthlyCharges": charges, "TenureMonths": tenure, "SupportTickets": tickets,
            "PaymentMethod": payment, "SatisfactionScore": satisfaction, "UsageHoursPerWeek": usage,
        }
        try:
            result = predictor.predict(record)
            st.divider()
            c1, c2 = st.columns([1, 2])
            with c1:
                st.metric("Churn Probability", f"{result.churn_probability:.1%}")
                badge_cls = "churn" if "Churn" in result.prediction else "stay"
                st.markdown(f'<span class="badge {badge_cls}">{result.prediction}</span>', unsafe_allow_html=True)
            with c2:
                st.progress(result.churn_probability)
                st.caption(f"Model used: {result.model_used}")
        except InvalidCustomerInputError as e:
            st.error(str(e))

# ---------------------------------------------------------------
with tab_insights:
    df = get_dataset()
    if df is None:
        st.warning("No dataset found. Run the training pipeline first.")
    else:
        churn_rate = df["Churn"].mean() * 100
        m1, m2, m3, m4 = st.columns(4)
        for col, k, l in zip(
            [m1, m2, m3, m4],
            [f"{len(df):,}", f"{churn_rate:.1f}%", f"{df['SatisfactionScore'].mean():.1f}", f"{df['TenureMonths'].mean():.0f}mo"],
            ["Total customers", "Overall churn rate", "Avg. satisfaction", "Avg. tenure"],
        ):
            col.markdown(f'<div class="metric-card"><div class="k">{k}</div><div class="l">{l}</div></div>', unsafe_allow_html=True)

        st.divider()
        chart_files = [
            "churn_pie_chart.png", "subscription_wise_churn.png",
            "correlation_heatmap.png", "monthly_charges_histogram.png",
            "boxplot_charges_by_churn.png", "scatter_tenure_vs_charges.png",
        ]
        cols = st.columns(2)
        for i, fname in enumerate(chart_files):
            path = IMAGES_DIR / fname
            if path.exists():
                cols[i % 2].image(str(path), use_container_width=True)

# ---------------------------------------------------------------
with tab_models:
    predictor = get_predictor()
    if predictor.comparison_table is not None:
        st.subheader("Model comparison")
        st.dataframe(predictor.comparison_table, use_container_width=True, hide_index=True)

    st.divider()
    cols = st.columns(2)
    for i, fname in enumerate(["model_comparison.png", "feature_importance.png"]):
        path = IMAGES_DIR / fname
        if path.exists():
            cols[i % 2].image(str(path), use_container_width=True)
