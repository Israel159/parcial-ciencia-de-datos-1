import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Churn Predictor", layout="centered")

st.title("Prediccion de Churn - Telco Customer")
st.write("Examen Parcial 2026-1 - Redes Neuronales Perceptron Multicapa")
st.write("---")

# Ruta a modelos
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "modelo_final"

st.write("Buscando modelos en: " + str(MODEL_DIR))

# Verificar archivos existen
archivos_requeridos = [
    "preprocessor.pkl",
    "input_columns.pkl", 
    "modelo_sklearn_final.pkl"
]

faltantes = []
for f in archivos_requeridos:
    if not (MODEL_DIR / f).exists():
        faltantes.append(f)

if faltantes:
    st.error("Faltan archivos en modelo_final/: " + ", ".join(faltantes))
    st.stop()

# Cargar modelos
try:
    preprocessor = joblib.load(MODEL_DIR / "preprocessor.pkl")
    input_columns = joblib.load(MODEL_DIR / "input_columns.pkl")
    modelo_sklearn = joblib.load(MODEL_DIR / "modelo_sklearn_final.pkl")
    st.write("Modelos cargados correctamente")
except Exception as e:
    st.error("Error cargando modelos: " + str(e))
    st.stop()

# Formulario
st.subheader("Datos del cliente")

gender = st.selectbox("Genero", ["Female", "Male"])
senior_citizen = st.selectbox("Adulto mayor", [0, 1])
partner = st.selectbox("Tiene pareja", ["Yes", "No"])
dependents = st.selectbox("Tiene dependientes", ["Yes", "No"])
tenure = st.number_input("Antiguedad (meses)", 0, 72, 12)
phone_service = st.selectbox("Servicio telefonico", ["Yes", "No"])
multiple_lines = st.selectbox("Lineas multiples", ["Yes", "No", "No phone service"])
internet_service = st.selectbox("Internet", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Seguridad online", ["Yes", "No", "No internet service"])
online_backup = st.selectbox("Respaldo online", ["Yes", "No", "No internet service"])
device_protection = st.selectbox("Proteccion dispositivo", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("Soporte tecnico", ["Yes", "No", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract = st.selectbox("Contrato", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Facturacion electronica", ["Yes", "No"])
payment_method = st.selectbox("Metodo de pago", [
    "Electronic check", "Mailed check", 
    "Bank transfer (automatic)", "Credit card (automatic)"
])
monthly_charges = st.number_input("Cargo mensual ($)", 0.0, 120.0, 70.0, 0.1)
total_charges = st.number_input("Cargo total ($)", 0.0, 9000.0, 800.0, 0.1)

st.write("---")

if st.button("PREDECIR CHURN"):

    datos = {
        'gender': gender, 'SeniorCitizen': senior_citizen,
        'Partner': partner, 'Dependents': dependents,
        'tenure': tenure, 'PhoneService': phone_service,
        'MultipleLines': multiple_lines, 'InternetService': internet_service,
        'OnlineSecurity': online_security, 'OnlineBackup': online_backup,
        'DeviceProtection': device_protection, 'TechSupport': tech_support,
        'StreamingTV': streaming_tv, 'StreamingMovies': streaming_movies,
        'Contract': contract, 'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges
    }

    df = pd.DataFrame([datos])

    # Feature engineering
    df['avg_charge_per_month'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['is_new_customer'] = (df['tenure'] <= 6).astype(int)
    df['long_term_contract'] = df['Contract'].isin(['One year', 'Two year']).astype(int)

    service_cols = ['PhoneService','OnlineSecurity','OnlineBackup',
                    'DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
    for col in service_cols:
        df[col + '_bin'] = df[col].replace({'Yes':1, 'No':0, 
                                            'No internet service':0, 'No phone service':0})
    df['service_count'] = df[[c + '_bin' for c in service_cols]].sum(axis=1)
    df['electronic_monthly_risk'] = (
        (df['PaymentMethod'] == 'Electronic check') & 
        (df['Contract'] == 'Month-to-month')
    ).astype(int)

    df = df[input_columns]
    X_proc = preprocessor.transform(df)

    proba = float(modelo_sklearn.predict_proba(X_proc)[0][1])
    pred = modelo_sklearn.predict(X_proc)[0]

    st.write("---")
    st.write("RESULTADO:")
    st.write("Probabilidad de Churn: " + str(round(proba*100, 1)) + "%")
    st.write("Prediccion: " + ("CHURN" if pred == 1 else "NO CHURN"))

    if proba >= 0.7:
        st.write("ALTO RIESGO: Contactar al cliente inmediatamente.")
    elif proba >= 0.5:
        st.write("RIESGO MODERADO: Monitorear y ofrecer upgrade.")
    else:
        st.write("BAJO RIESGO: Cliente estable.")

    st.write("---")
    st.write("Factores detectados:")
    if df['is_new_customer'].iloc[0] == 1:
        st.write("- Cliente nuevo")
    if df['electronic_monthly_risk'].iloc[0] == 1:
        st.write("- Pago electronico + mes a mes")
    if df['long_term_contract'].iloc[0] == 0:
        st.write("- Sin contrato largo plazo")
    if df['service_count'].iloc[0] <= 2:
        st.write("- Pocos servicios")
