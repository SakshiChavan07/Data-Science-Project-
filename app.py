import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Load the trained ML model ---
try:
    model = pickle.load(open('Project.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- Load the dataset ---
try:
    data = pd.read_csv('Heart_Attack_Risk_Levels_Dataset.csv')  # replace 'your_dataset.csv' with your actual filename
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# --- Frontend UI ---
st.title('❤️ Heart Attack Risk Predictor')

# Ideal values based on dataset analysis or medical standards
ideal_age = 40
ideal_heart_rate = 70
ideal_blood_sugar = 90

# --- User input ---
st.header("📝 Enter your details")
age = st.number_input('Enter Age', min_value=1, max_value=120, value=30)
heart_rate = st.number_input('Enter Heart Rate (bpm)', min_value=30, max_value=200, value=75)
blood_sugar = st.number_input('Enter Blood Sugar (mg/dL)', min_value=50, max_value=200, value=100)

# --- Predict button ---
if st.button('🔮 Predict Risk'):
    # Prepare input for prediction
    input_features = np.array([[age, heart_rate, blood_sugar]])

    # Make prediction
    pred = model.predict(input_features)

    # Show prediction result
    if pred[0] == 1:
        st.error('⚠️ **High Risk of Heart Attack!** 😟')
    else:
        st.success('✅ **Low Risk! Stay Healthy!** 💪')

    # --- Comparison plot ---
    st.subheader('🔍 Comparison of Your Input vs. Ideal Values')

    categories = ['Age', 'Heart Rate', 'Blood Sugar']
    user_values = [age, heart_rate, blood_sugar]
    ideal_values = [ideal_age, ideal_heart_rate, ideal_blood_sugar]

    y_pos = np.arange(len(categories))
    bar_width = 0.35

    fig, ax = plt.subplots()
    ax.barh(y_pos, user_values, bar_width, label='Your Values', color='lightblue')
    ax.barh(y_pos + bar_width, ideal_values, bar_width, label='Ideal Values', color='lightgreen')

    ax.set_yticks(y_pos + bar_width / 2)
    ax.set_yticklabels(categories)
    ax.set_xlabel('Values')
    ax.set_title('Comparison of Your Input vs. Ideal Values')
    ax.legend()

    st.pyplot(fig)

    # --- Fun emoji summary ---
    if pred[0] == 1:
        st.image('https://i.imgur.com/dDduCZ9.png', use_column_width=True)
    else:
        st.image('https://i.imgur.com/4u9JTxZ.png', use_column_width=True)

# Optional: Show Dataset Preview
st.sidebar.title('📄 Dataset Preview')
if st.sidebar.checkbox('Show Raw Dataset'):
    st.sidebar.dataframe(data)
