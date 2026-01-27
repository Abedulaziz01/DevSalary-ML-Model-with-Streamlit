import streamlit as st
import pickle as pkl
import numpy as np


def load_model():
    with open("salary_model.pkl", "rb") as file:
        model = pkl.load(file)
    return model


data = load_model()

model = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]


def show_predict_page():
    st.title("Software Developer Salary Prediction")

    st.write("### Enter your information")

    country = st.selectbox(
        "Country",
        le_country.classes_
    )

    education = st.selectbox(
        "Education Level",
        le_education.classes_
    )

    experience = st.slider(
        "Years of Experience"
    )

    if st.button("Calculate Salary"):
        # Encode inputs (VERY IMPORTANT)
        country_encoded = le_country.transform([country])[0]
        education_encoded = le_education.transform([education])[0]

        # Create input exactly like training X
        X_input = np.array([[country_encoded, education_encoded, experience]])

        salary = model.predict(X_input)[0]

        st.success(f"Estimated Salary: ${salary:,.0f}")