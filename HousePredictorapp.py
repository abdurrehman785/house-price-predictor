import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="🏡✨ Cutey House Price Predictor", page_icon="🌸")

# Cute title and emoji
st.title("🏡💖 House Price Predictor ✨")
st.markdown("Welcome to your price predictor! Enter your cute house details below, and I'll guess the price for you! 🧚‍♀️🌷")

# Load data
df = pd.read_csv('C:\\Users\\Admin\\Desktop\\price prediction\\zameen.csv', sep='|')

# Prepare data
x = df[['city', 'location', 'bedrooms', 'baths', 'size']]
y = df['price']
x_encoded = pd.get_dummies(x, columns=['city', 'location'], drop_first=True)

model = LinearRegression()
model.fit(x_encoded, y)

# Get unique options for city and location to help user
cities = df['city'].unique().tolist()
locations = df['location'].unique().tolist()

# User inputs
st.header("✨ Your Cute House Details ✨")
user_city = st.selectbox("🌆 Choose your city:", cities)
user_location = st.selectbox("📍 Choose your location:", locations)
user_bedrooms = st.number_input("🛏️ Number of bedrooms:", min_value=1, step=1)
user_baths = st.number_input("🛁 Number of bathrooms:", min_value=1, step=1)
user_size = st.number_input("📐 Size of the house (sq ft):", min_value=1)

# Predict button
if st.button("💫 Predict My Cute House Price!"):
    # Prepare user input
    user_house_df = pd.DataFrame({
        'city': [user_city],
        'location': [user_location],
        'bedrooms': [user_bedrooms],
        'baths': [user_baths],
        'size': [user_size]
    })
    
    user_house_encoded = pd.get_dummies(user_house_df, columns=['city', 'location'], drop_first=True)
    
    # Add missing columns
    for col in x_encoded.columns:
        if col not in user_house_encoded.columns:
            user_house_encoded[col] = 0

    # Ensure columns order
    user_house_encoded = user_house_encoded[x_encoded.columns]
    
    predicted_price = model.predict(user_house_encoded)[0]
    
    st.success(f"✨🎉 Your amazings house is estimated to cost: **PKR {predicted_price:,.0f}** 💖🏡✨")

    st.balloons()
