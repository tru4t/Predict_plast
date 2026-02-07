import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

data = {
    'coffee_cups': [0, 5, 10, 2, 7, 1],
    'bags': [1, 10, 20, 5, 15, 2],
    'takeout': [0, 3, 7, 1, 5, 0],
    'waste_kg': [5.2, 25.4, 55.1, 12.8, 38.2, 7.5] # Target variable
}

df = pd.DataFrame(data)
X = df[['coffee_cups', 'bags', 'takeout']]
y = df['waste_kg']

model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)

joblib.dump(model, 'eco_model.pkl')

from google.colab import files
files.download('eco_model.pkl')

import streamlit as st
import joblib
import numpy as np

model = joblib.load('eco_model.pkl')

st.title("🌱 Plastic-Free Community Eco-Score")
st.write("Predict your annual plastic footprint and see how to improve!")

# User Inputs
cups = st.slider("Weekly takeaway coffee cups", 0, 20, 5)
bags = st.slider("Weekly plastic grocery bags", 0, 30, 10)
takeout = st.slider("Weekly takeout meals", 0, 14, 3)

if st.button("Predict My Footprint"):
    # Make prediction
    prediction = model.predict([[cups, bags, takeout]])
    result = round(prediction[0], 2)

    st.header(f"Your Predicted Impact: {result} kg / year")

    # Simple Logic for Feedback
    if result > 30:
        st.warning("That's quite high! Try switching to a reusable bottle and tote bags.")
    else:
        st.success("Great job! You are below the average plastic consumer.")
