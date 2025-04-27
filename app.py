import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load your saved model
try:
    model = pickle.load(open('Project (2).pkl', 'rb'))
    st.success('✅ Model loaded successfully!')
except Exception as e:
    st.error(f"Error loading model: {e}")

# Frontend UI
st.title('❤️ Heart Attack Risk Predictor')
st.write('---')

# Ideal healthy values (you can tweak these)
ideal_age = 40
ideal_heart_rate = 70
ideal_blood_sugar = 90

# User input
st.header('📝 Enter your details below:')
age = st.number_input('Enter Age (years)', min_value=1, max_value=120, value=30)
heart_rate = st.number_input('Enter Heart Rate (bpm)', min_value=30, max_value=200, value=75)
blood_sugar = st.number_input('Enter Blood Sugar (mg/dL)', min_value=50, max_value=300, value=100)

# Predict Button
if st.button('🔮 Predict Risk'):
    input_data = [[age, heart_rate, blood_sugar]]
    
    # Make prediction
    pred = model.predict(input_data)

    if pred[0] == 1:
        st.error('⚠️ **High Risk of Heart Attack! Please consult your doctor.** 😟')
    else:
        st.success('✅ **Low Risk! Stay Healthy!** 💪')

    st.write('---')

    # Show comparison with ideal values
    st.subheader('📊 Your Inputs vs Ideal Health Values')

    categories = ['Age', 'Heart Rate', 'Blood Sugar']
    user_values = [age, heart_rate, blood_sugar]
    ideal_values = [ideal_age, ideal_heart_rate, ideal_blood_sugar]

    y_pos = np.arange(len(categories))
    bar_width = 0.35

    fig, ax = plt.subplots()

    ax.barh(y_pos, user_values, bar_width, label='Your Values', color='skyblue')
    ax.barh(y_pos + bar_width, ideal_values, bar_width, label='Ideal Values', color='lightgreen')

    ax.set_yticks(y_pos + bar_width / 2)
    ax.set_yticklabels(categories)
    ax.set_xlabel('Values')
    ax.set_title('Health Parameters Comparison')
    ax.legend()

    st.pyplot(fig)

    # Add a fun image depending on prediction
    if pred[0] == 1:
        st.image('https://i.imgur.com/dDduCZ9.png', use_column_width=True)
    else:
        st.image('https://i.imgur.com/4u9JTxZ.png', use_column_width=True)

# Footer
st.write('---')
st.caption('Built with ❤️ using Streamlit')
