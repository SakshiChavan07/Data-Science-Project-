import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load your saved model (uncomment when you have the model file)
# model = pickle.load(open('Project.pkl', 'rb'))

# Frontend UI
st.title('❤️ Heart Attack Risk Predictor')

# Ideal values (these can be adjusted based on what is considered 'normal' or 'ideal' for a healthy individual)
ideal_age = 40
ideal_heart_rate = 70
ideal_blood_sugar = 90

# User input
age = st.number_input('Enter Age', min_value=1, max_value=120, value=30)
heart_rate = st.number_input('Enter Heart Rate (bpm)', min_value=30, max_value=200, value=75)
blood_sugar = st.number_input('Enter Blood Sugar (mg/dL)', min_value=50, max_value=200, value=100)

# Prediction button
if st.button('🔮 Predict Risk'):
    # Prediction (uncomment when the model is loaded)
    # pred = model.predict([[age, heart_rate, blood_sugar]])

    # For demo, we are assuming a prediction (1 = High Risk, 0 = Low Risk)
    pred = [1]  # Change this to test different cases

    if pred[0] == 1:
        st.error('⚠️ **High Risk of Heart Attack!** 😟')
    else:
        st.success('✅ **Low Risk! Stay Healthy!** 💪')

    # Show Bar Comparison
    st.subheader('🔍 Comparison of Your Input vs. Ideal Values')

    # Set up data for comparison
    categories = ['Age', 'Heart Rate', 'Blood Sugar']
    user_values = [age, heart_rate, blood_sugar]
    ideal_values = [ideal_age, ideal_heart_rate, ideal_blood_sugar]

    # Create a horizontal bar chart to compare entered values with ideal values
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

    # Display the plot
    st.pyplot(fig)

    # Show a fun emoji-based summary
    if pred[0] == 1:
        st.image('https://i.imgur.com/dDduCZ9.png', use_column_width=True)
    else:
        st.image('https://i.imgur.com/4u9JTxZ.png', use_column_width=True)
