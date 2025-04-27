import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the trained model
try:
    model = pickle.load(open('Project (2).pkl', 'rb'))
    st.success('✅ Model loaded successfully!')
except Exception as e:
    st.error(f"Error loading model: {e}")

# Frontend UI
st.title('❤️ Heart Attack Risk Predictor')
st.write('---')

# Ideal healthy values (you can tweak these)
ideal_heart_rate = (60, 100)  # Ideal Heart Rate range
ideal_blood_sugar = (70, 110)  # Ideal Blood Sugar range

# User input
st.header('📝 Enter your details below:')
age = st.number_input('Enter Age (years)', min_value=1, max_value=120, value=30)
gender = st.selectbox('Select Gender', ['Male', 'Female'])
heart_rate = st.number_input('Enter Heart Rate (bpm)', min_value=30, max_value=200, value=75)
blood_sugar = st.number_input('Enter Blood Sugar (mg/dL)', min_value=50, max_value=300, value=100)

# Create input data for prediction - only include the features expected by the model
input_data = [[age, heart_rate, blood_sugar]]  # Only 3 features: age, heart_rate, and blood_sugar

# Feature Scaling (if the model was trained with scaled features)
scaler = StandardScaler()
scaled_input_data = scaler.fit_transform(input_data)

# Predict Button
if st.button('🔮 Predict Risk'):
    try:
        # Predict using the model
        pred = model.predict(scaled_input_data)
        result = pred[0]  # Get the prediction (0 = low risk, 1 = high risk)

        # Show result
        if result == 1:
            st.error('⚠️ **High Risk of Heart Attack! Please consult your doctor.** 😟')
        else:
            st.success('✅ **Low Risk! Stay Healthy!** 💪')
    except Exception as e:
        st.error(f"Error during prediction: {e}")

    st.write('---')

    # Show comparison with ideal values (for illustration purposes)
    st.subheader('📊 Your Inputs vs Ideal Health Values')

    categories = ['Heart Rate', 'Blood Sugar']
    user_values = [heart_rate, blood_sugar]
    ideal_values = [ideal_heart_rate, ideal_blood_sugar]

    y_pos = np.arange(len(categories))
    bar_width = 0.35

    # Create plot only if the data is valid
    try:
        fig, ax = plt.subplots(figsize=(8, 4))  # Set the figure size to avoid rendering issues
        ax.barh(y_pos, user_values, bar_width, label='Your Values', color='skyblue')
        ax.barh(y_pos + bar_width, ideal_values, bar_width, label='Ideal Values', color='lightgreen')

        ax.set_yticks(y_pos + bar_width / 2)
        ax.set_yticklabels(categories)
        ax.set_xlabel('Values')
        ax.set_title('Health Parameters Comparison')
        ax.legend()

        st.pyplot(fig)  # Display the plot
    except Exception as e:
        st.error(f"Error during plot generation: {e}")

# Footer
st.write('---')
st.caption('Built with ❤️ using Streamlit')
