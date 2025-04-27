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

# Frontend UI
st.title('❤️ Heart Attack Risk Predictor')
st.write('---')

# Ideal healthy ranges
ideal_heart_rate_range = (60, 100)         # bpm
ideal_systolic_bp_range = (90, 120)         # mmHg
ideal_diastolic_bp_range = (60, 80)         # mmHg
ideal_blood_sugar_range = (70, 110)         # mg/dL

# User input
st.header('📝 Enter Patient Details:')

age = st.number_input('Age (years)', min_value=1, max_value=120, value=30)

gender = st.selectbox('Gender', ['Male', 'Female'])  # Assuming gender can be encoded later

heart_rate = st.number_input('Heart Rate (bpm)', min_value=30, max_value=200, value=75)

systolic_bp = st.number_input('Systolic Blood Pressure (mmHg)', min_value=80, max_value=200, value=120)

diastolic_bp = st.number_input('Diastolic Blood Pressure (mmHg)', min_value=40, max_value=130, value=80)

blood_sugar = st.number_input('Blood Sugar (mg/dL)', min_value=50, max_value=300, value=100)

# Encoding gender
gender_encoded = 1 if gender == 'Male' else 0

# Predict Button
if st.button('🔮 Predict Risk'):
    # Prepare input according to training
    input_data = np.array([[age, gender_encoded, heart_rate, systolic_bp, diastolic_bp, blood_sugar]])

    # Make prediction
    pred = model.predict(input_data)

    # Show predictions
    st.subheader('📢 Prediction Results')

    # Assume your model returns Risk_Level based on prediction (you can map accordingly)
    if pred[0] == 0:
        result = 'Normal'
        risk_level = 'Low Risk'
        recommendation = 'Maintain a healthy lifestyle. Regular checkups recommended.'
        st.success('✅ Patient is likely Normal. Stay healthy! 💪')
    else:
        result = 'Abnormal'
        risk_level = 'High Risk'
        recommendation = 'Immediate medical consultation recommended. 🏥'
        st.error('⚠️ High Risk detected! Immediate action needed.')

    # Display prediction details
    st.write(f'**Result:** {result}')
    st.write(f'**Risk Level:** {risk_level}')
    st.write(f'**Recommendation:** {recommendation}')

    st.write('---')

    # Show comparison with ideal ranges
    st.subheader('📊 Your Inputs vs Ideal Health Ranges')

    comparison_table = pd.DataFrame({
        'Parameter': ['Heart Rate (bpm)', 'Systolic BP (mmHg)', 'Diastolic BP (mmHg)', 'Blood Sugar (mg/dL)'],
        'Your Input': [heart_rate, systolic_bp, diastolic_bp, blood_sugar],
        'Ideal Range': [
            f"{ideal_heart_rate_range[0]}-{ideal_heart_rate_range[1]}",
            f"{ideal_systolic_bp_range[0]}-{ideal_systolic_bp_range[1]}",
            f"{ideal_diastolic_bp_range[0]}-{ideal_diastolic_bp_range[1]}",
            f"{ideal_blood_sugar_range[0]}-{ideal_blood_sugar_range[1]}"
        ]
    })

    st.table(comparison_table)

# Footer
st.write('---')
st.caption('Built with ❤️ using Streamlit')
