import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
import base64

# ⬇️ CALL THIS FUNCTION TO SET BACKGROUND
def set_background(image_file):
    with open(image_file, "rb") as img:
        b64 = base64.b64encode(img.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpg;base64,{b64}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

# ✅ APPLY BACKGROUND IMAGE
set_background("back.jpg")

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("car_data.csv")
    df = df[df['Price'] != 'Ask For Price']
    df['Price'] = df['Price'].str.replace(',', '').astype(int)
    df['kms_driven'] = df['kms_driven'].str.replace(' kms', '').str.replace(',', '')
    df['kms_driven'] = pd.to_numeric(df['kms_driven'], errors='coerce')
    df.dropna(inplace=True)
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df.dropna(inplace=True)
    df['year'] = df['year'].astype(int)
    df['car_model'] = df['name'].str.split().str.slice(0,2).str.join(" ")
    df.drop('name', axis=1, inplace=True)
    df = df[['car_model', 'company', 'year', 'Price', 'kms_driven', 'fuel_type']]
    return df

df = load_data()

@st.cache_resource
def train_model():
    dff = pd.get_dummies(df.drop('Price', axis=1), drop_first=True)
    X = dff
    y = df['Price']
    model = RandomForestRegressor()
    model.fit(X, y)
    return model, X.columns

model, feature_names = train_model()

st.title("🚗 Car Price Predictor App")

car_model = st.selectbox("Select Car Model", sorted(df['car_model'].unique()))
company = st.selectbox("Select Company", sorted(df['company'].unique()))
year = st.selectbox("Select Year", sorted(df['year'].unique(), reverse=True))
fuel = st.selectbox("Fuel Type", sorted(df['fuel_type'].unique()))
kms_driven = st.number_input("Kilometers Driven", min_value=100, step=100)

if st.button("Predict Price"):
    input_data = pd.DataFrame([[car_model, company, year, kms_driven, fuel]], 
        columns=['car_model', 'company', 'year', 'kms_driven', 'fuel_type'])
    input_encoded = pd.get_dummies(input_data)
    input_encoded = input_encoded.reindex(columns=feature_names, fill_value=0)
    prediction = model.predict(input_encoded)[0]
    st.success(f"💰 Estimated Price: ₹ {int(prediction):,}")
