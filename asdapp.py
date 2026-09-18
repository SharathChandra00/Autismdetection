import streamlit as st
import pandas as pd
from asdmodel import ASDModel, SCALERS, CLASSIFIERS

st.set_page_config(page_title="ASD Predictor", layout="centered")

# Hide Streamlit Deploy button, top header, main menu, and footer
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stHeader"] {visibility: hidden;}
    .stDeployButton {display: none;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title(" ASD Detection")

uploaded_file = st.file_uploader(" Upload ASD Dataset (CSV)", type=["csv"])
scaler_choice = st.sidebar.selectbox(" Choose Scaler", list(SCALERS.keys()))
clf_choice = st.sidebar.selectbox(" Choose Classifier", list(CLASSIFIERS.keys()))

model = ASDModel()

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(" Data Preview")
    st.dataframe(df.head())

    target_col = df.columns[-1]  
    X, y = model.preprocess(df, target_col)
    model.train(X, y, scaler_choice, clf_choice)

    st.subheader(" Enter Feature Values for Prediction")
    input_data = {}
    for col in model.X_columns:
        if col in model.label_encoders:
            options = list(model.label_encoders[col].classes_)
            input_data[col] = st.selectbox(f"{col}", options)
        else:
            input_data[col] = st.number_input(f"{col}", value=0.0)

    if st.button(" Predict ASD"):
        try:
            input_array = model.encode_input(input_data)
            prediction = model.predict(input_array)
            proba = model.predict_proba(input_array)

            st.success(f" Prediction: {'ASD Detected' if prediction == 1 else 'No ASD'}")
            if proba is not None:
                st.info(f" Probability of ASD: {proba:.4f}")
        except Exception as e:
            st.error(f" Error: {str(e)}")
