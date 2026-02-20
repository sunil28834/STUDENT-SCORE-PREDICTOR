import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("model.pkl","rb"))

st.title("🎓 Student Score Predictor")

hours = st.number_input("Hours Studied", min_value=0.0)
sleep = st.number_input("Sleep Hours", min_value=0.0)
att = st.number_input("Attendance (%)", min_value=0.0)

if st.button("Predict"):
    result = model.predict([[hours,sleep,att]])
    st.success(f"Predicted Score: {result[0]:.2f}")
