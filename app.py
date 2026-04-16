import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# Save original data for graphs
original_data = data.copy()

# Normalize
mean = data.mean()
std = data.std()
data = (data - mean) / std

# Convert to array
data = np.array(data)

# Split features & target
Y = data[:, 0:1]
X = data[:, 1:]

# Add bias
ones = np.ones((X.shape[0], 1))
X = np.concatenate((ones, X), axis=1)

# Train model (Normal Equation)
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

user_input = [lot, bed, bath, stories, garage]

# ==============================
# PREDICTION
# ==============================
if st.sidebar.button("Predict Price"):

    user_input = np.array(user_input)

    # Normalize input
    user_input = (user_input - mean[1:].values) / std[1:].values

    # Add bias
    user_input = np.insert(user_input, 0, 1)
    user_input = user_input.reshape(1, -1)

    # Predict
    prediction = user_input @ theta

    # Denormalize
    predicted_price = prediction[0][0] * std[0] + mean[0]

    # Safety fix
    predicted_price = max(0, predicted_price)

    st.subheader("💰 Predicted Price")
    st.success(f"₹ {predicted_price:,.2f}")

# ==============================
# DATASET PREVIEW
# ==============================
st.subheader("📊 Dataset Preview")
st.write(original_data.head())

# ==============================
# GRAPHS
# ==============================
st.subheader("📈 Data Distribution")

fig, ax = plt.subplots()
original_data.hist(ax=ax)
st.pyplot(fig)

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