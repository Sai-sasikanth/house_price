import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------------------
# Page Configuration and Light Blue Background Styling
# -------------------------------------------
st.set_page_config(page_title=" House Price Predictor", layout="centered")

st.markdown(
    """
    <style>
    .stApp {
        background-color: #3385ff;  /* Light blue background */
    }
    .main {
        padding: 2rem;
        font-family: "Segoe UI", sans-serif;
    }
    h1, h2, h3 {
        color: #2C3E50;
    }
    .stButton>button {
        background-color: white;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5em 2em;
    }
    .stSuccess {
        background-color: #D4EFDF;
        color: #1E8449;
        font-size: 18px;
        font-weight: bold;
        padding: 10px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------
# Load California Housing Dataset
# -------------------------------------------
housing = fetch_california_housing()
data = pd.DataFrame(housing.data, columns=housing.feature_names)
data['PRICE'] = housing.target

X = data.drop('PRICE', axis=1)
y = data['PRICE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------------------
# Streamlit App UI
# -------------------------------------------
st.title(" California House Price Prediction App")
st.markdown("### Fill in the house details below to predict the price")

# Input form
st.markdown("####  Input Features")

def user_input_features():
    cols = st.columns(2)
    input_data = {}

    for i, feature in enumerate(X.columns):
        with cols[i % 2]:
            input_data[feature] = st.number_input(
                f"{feature}",
                min_value=float(data[feature].min()),
                max_value=float(data[feature].max()),
                value=float(data[feature].mean()),
                step=0.1
            )
    return pd.DataFrame([input_data])

input_df = user_input_features()

# Predict button
if st.button("🔍 Predict Price"):
    prediction = model.predict(input_df)
    st.success(f" Estimated House Price: **${prediction[0] * 100000:.2f}**")

# -------------------------------------------
# Sidebar: EDA & Model Evaluation
# -------------------------------------------
st.sidebar.header("📊 Exploratory Data & Evaluation")

# Correlation Heatmap
if st.sidebar.checkbox("📌 Show Correlation Heatmap"):
    st.sidebar.markdown("Feature correlation matrix")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.sidebar.pyplot(fig)

# Model Performance
if st.sidebar.checkbox("📈 Show Model Evaluation"):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.sidebar.markdown(f"**📉 Mean Squared Error:** `{mse:.2f}`")
    st.sidebar.markdown(f"**📊 R² Score:** `{r2:.2f}`")

    fig2, ax2 = plt.subplots()
    ax2.scatter(y_test, y_pred, edgecolors=(0, 0, 0))
    ax2.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
    ax2.set_xlabel("Actual Prices")
    ax2.set_ylabel("Predicted Prices")
    ax2.set_title("Actual vs Predicted House Prices")
    st.sidebar.pyplot(fig2)

# Footer
st.markdown("""
    <hr style='border: 1px solid #ccc;'>
""", unsafe_allow_html=True)
