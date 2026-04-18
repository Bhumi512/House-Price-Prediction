import streamlit as st
import numpy as np
import pandas as pd

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="House Price Predictor", layout="wide")

# ==============================
# TITLE
# ==============================
st.title("🏠 House Price Prediction App")
st.markdown("### Predict house prices using Machine Learning")

# ==============================
# LOAD DATA
# ==============================
data = pd.read_csv("dataset.csv")

# Drop unnecessary columns
data = data.drop(['Unnamed: 0','driveway','recroom','fullbase','airco','prefarea','gashw'], axis=1)

# Save original data
original_data = data.copy()

# ==============================
# SEPARATE FEATURES & TARGET
# ==============================
target_col = data.columns[0]   # price column
feature_cols = data.columns[1:]

# Normalize
mean = data.mean()
std = data.std()

data_norm = (data - mean) / std

# Convert to array
data_np = np.array(data_norm)

Y = data_np[:, 0:1]
X = data_np[:, 1:]

# Add bias
ones = np.ones((X.shape[0], 1))
X = np.concatenate((ones, X), axis=1)

# Train model
theta = np.linalg.inv(X.T @ X) @ X.T @ Y

st.success("✅ Model Trained Successfully!")

# ==============================
# SIDEBAR INPUTS
# ==============================
st.sidebar.header("📥 Enter House Details")

lot = st.sidebar.slider("Lot Size", 1000, 10000, 3000)
bed = st.sidebar.slider("Bedrooms", 1, 5, 3)
bath = st.sidebar.slider("Bathrooms", 1, 4, 2)
stories = st.sidebar.slider("Stories", 1, 3, 2)
garage = st.sidebar.slider("Garage", 0, 3, 1)

user_input = np.array([lot, bed, bath, stories, garage])

# ==============================
# PREDICTION
# ==============================
if st.sidebar.button("Predict Price"):

    # Normalize input (ONLY features)
    user_input_norm = (user_input - mean[feature_cols].values) / std[feature_cols].values

    # Add bias
    user_input_norm = np.insert(user_input_norm, 0, 1)
    user_input_norm = user_input_norm.reshape(1, -1)

    # Predict
    prediction = user_input_norm @ theta

    # Denormalize using TARGET column
    predicted_price = prediction[0][0] * std[target_col] + mean[target_col]

    predicted_price = max(0, predicted_price)

    st.subheader("💰 Predicted Price")
    st.success(f"₹ {predicted_price:,.2f}")

# ==============================
# DATASET PREVIEW
# ==============================
st.subheader("📊 Dataset Preview")
st.write(original_data.head())

# ==============================
# CORRELATION
# ==============================
st.subheader("🔗 Feature Correlation")
st.write(original_data.corr())

# ==============================
# MODEL INFO
# ==============================
st.subheader("🧠 About Model")

st.write("""
This project uses **Linear Regression** to predict house prices.

### Features used:
- Lot Size
- Bedrooms
- Bathrooms
- Stories
- Garage

### Steps:
1. Data Cleaning
2. Normalization
3. Model Training (Normal Equation)
4. Prediction

### Note:
This model provides approximate predictions and may not reflect real market prices.
""")
