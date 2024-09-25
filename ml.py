import streamlit as st
import joblib
import numpy as np

@st.cache_data
def load_model():
    model_file_path = 'seattle_weather_decision_tree_model.pkl'
    try:
        model = joblib.load(model_file_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

def predict_weather(precipitation, temp_max, temp_min, wind):
    input_data = np.array([[precipitation, temp_max, temp_min, wind]])
    prediction = model.predict(input_data)
    return prediction

st.title("Seattle Weather Prediction App")

st.write("Enter the following weather parameters to predict the type of weather:")


precipitation = st.slider("Precipitation (in mm)", 0.0, 500.0, 1.0)
temp_max = st.slider("Max Temperature (in °C)", -20.0, 50.0, 10.0)
temp_min = st.slider("Min Temperature (in °C)", -30.0, 50.0, 5.0)
wind = st.slider("Wind Speed (in m/s)", 0.0, 100.0, 2.0)


if st.button("Predict Weather"):
    if model:
        prediction = predict_weather(precipitation, temp_max, temp_min, wind)
        
        
        weather_labels = {0: "Drizzle", 1: "Fog", 2: "Rain", 3: "Snow", 4: "Sun"}
        st.success(f"Predicted Weather: {weather_labels.get(prediction[0], 'Unknown')}")
    else:
        st.error("Model is not loaded. Please check the model file.")

@st.cache_data
def load_data():
  
    return data
import streamlit as st

@st.cache_resource
def load_model():
    
    return model
