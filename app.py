# app.py

import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("tuned_decision_tree_model.pkl")

# Title
st.title("Social Media Usage and Emotional Well-Being Prediction")

st.write("Enter user details to predict emotional well-being.")

# User Inputs
age = st.number_input("Age", min_value=10, max_value=100, value=25)

gender = st.selectbox("Gender", ["Male", "Female"])

platform = st.selectbox(
    "Social Media Platform",
    ["Instagram", "Facebook", "Twitter", "WhatsApp", "Snapchat", "LinkedIn"]
)

daily_usage = st.number_input(
    "Daily Usage Time (minutes)",
    min_value=0,
    max_value=1440,
    value=120
)

posts_per_day = st.number_input(
    "Posts Per Day",
    min_value=0,
    max_value=100,
    value=5
)

likes_received = st.number_input(
    "Likes Received Per Day",
    min_value=0,
    max_value=10000,
    value=50
)

comments_received = st.number_input(
    "Comments Received Per Day",
    min_value=0,
    max_value=5000,
    value=20
)

messages_sent = st.number_input(
    "Messages Sent Per Day",
    min_value=0,
    max_value=5000,
    value=30
)

# Encoding
gender_map = {
    "Male": 1,
    "Female": 0
}

platform_map = {
    "Instagram": 0,
    "Facebook": 1,
    "Twitter": 2,
    "WhatsApp": 3,
    "Snapchat": 4,
    "LinkedIn": 5
}

gender_encoded = gender_map[gender]
platform_encoded = platform_map[platform]

# Create DataFrame
input_data = pd.DataFrame(
    [[
        age,
        gender_encoded,
        platform_encoded,
        daily_usage,
        posts_per_day,
        likes_received,
        comments_received,
        messages_sent
    ]],
    columns=[
        "Age",
        "Gender",
        "Platform",
        "Daily_Usage_Time (minutes)",
        "Posts_Per_Day",
        "Likes_Received_Per_Day",
        "Comments_Received_Per_Day",
        "Messages_Sent_Per_Day"
    ]
)

# Prediction
if st.button("Predict Emotion"):
    
    prediction = model.predict(input_data)

    emotion_map = {
        0: "Anxiety",
        1: "Boredom",
        2: "Happiness",
        3: "Neutral",
        4: "Sadness"
    }

    predicted_emotion = emotion_map.get(prediction[0], prediction[0])

    st.success(f"Predicted Emotional State: {predicted_emotion}")
