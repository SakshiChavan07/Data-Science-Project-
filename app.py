import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load your saved model
try:
    model = pickle.load(open('Project (2).pkl', 'rb'))
    st.success('✅ Model loaded successfully!')
except Exception as e:
    st.error(f"Error loading model: {e}")

# Frontend UI with a little emoji magic ✨
st.title('❤️ Heart Attack Risk Predictor')
st.write('---')

# Ideal healthy ranges with some emoji fun 🎉
ideal_heart_rate_range = (60, 100)         # bpm
ideal_blood_sugar_range = (70, 110)         # mg/dL

# User input with emojis to make it fun ✨
st.header('📝 Enter Patient Details:')

age = st.number_input('👩‍⚕️ Age (years)', min_value=1, max_value=120, value=00)
heart_rate = st.number_input('💓 Heart Rate (bpm)', min_value=30, max_value=200, value=00)
blood_sugar = st.number_input('🍬 Blood Sugar (mg/dL)', min_value=50, max_value=300, value=00)

# Prediction Button with a ✨ Predict emoji ✨
if st.button('🔮 Predict Risk'):
    # Ensure input data matches the model's expected features (only 3 features here)
    input_data = np.array([[age, heart_rate, blood_sugar]])

    try:
        # Make prediction
        pred = model.predict(input_data)

        # Prediction results and emoji-based outcome 🌟
        if pred[0] == 0:
            result = 'Normal ✅'
            risk_level = 'Low Risk 💚'
            recommendation = 'Maintain a healthy lifestyle. Regular checkups recommended. 💪'
            st.success(f'✅ **Risk Level:** {risk_level} \n**Result:** {result} 🏋️‍♀️ Stay Healthy!')
        else:
            result = 'Abnormal ⚠️'
            risk_level = 'High Risk 🔴'
            recommendation = 'Immediate medical consultation recommended. 🚑'
            st.error(f'⚠️ **Risk Level:** {risk_level} \n**Result:** {result} 🚨 Immediate action needed!')

        st.write(f'**Recommendation:** {recommendation}')

    except Exception as e:
        st.error(f"Error during prediction: {e}")

    # Display ideal ranges comparison in a cool table style 📊
    st.write('---')
    st.subheader('📊 Your Inputs vs Ideal Health Ranges')

    comparison_table = pd.DataFrame({
        'Parameter': ['Heart Rate (bpm)', 'Blood Sugar (mg/dL)'],
        'Your Input': [heart_rate, blood_sugar],
        'Ideal Range': [
            f"{ideal_heart_rate_range[0]}-{ideal_heart_rate_range[1]}",
            f"{ideal_blood_sugar_range[0]}-{ideal_blood_sugar_range[1]}"
        ]
    })

    st.table(comparison_table)

# Footer with some style and emojis ✨
st.write('---')
st.caption('Built with ❤️ using Streamlit and Python! 🚀')
