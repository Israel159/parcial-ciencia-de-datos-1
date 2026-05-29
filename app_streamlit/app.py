import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ───────────────────────────────────────────────
# CONFIGURACION BASICA
# ───────────────────────────────────────────────
st.set_page_config(page_title="Churn Predictor", layout="centered")

SEED = 42
np.random.seed(SEED)

# ───────────────────────────────────────────────
# RUTAS
# ───────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "modelo_final"

# ───────────────────────────────────────────────
# CARGAR MODELOS (con fallback si TensorFlow falla)
# ───────────────────────────────────────────────
@st.cache_resource
def cargar_modelos():
    preprocessor = joblib.load(MODEL_DIR / "preprocessor.pkl")
    input_columns = joblib.load(MODEL_DIR / "input_columns.pkl")
    modelo_sklearn = joblib.load(MODEL_DIR / "modelo_sklearn_final.pkl")

    # Intentar cargar Keras, si falla usamos solo sklearn
    try:
        import tensorflow as tf
        from tensorflow import keras
        tf.random.set_seed(SEED)
        modelo_keras = keras.models.load_model(MODEL_DIR / "modelo_keras_final.keras")
        return preprocessor, input_columns, modelo_keras, modelo_sklearn, True
    except Exception:
        return preprocessor, input_columns, None, modelo_sklearn, False

preprocessor, input_columns, modelo_keras, modelo_sklearn, keras_ok = cargar_modelos()

# ───────────────────────────────────────────────
# TITULO
# ───────────────────────────────────────────────
st.title("Prediccion de Churn - Telco Customer")
st.write("Examen Parcial 2026-1 - Redes Neuronales Perceptron Multicapa")
st.write("---")

# ───────────────────────────────────────────────
# FORMULARIO VERTICAL SIMPLE (sin columnas, sin form)
# ───────────────────────────────────────────────
st.subheader("Datos del cliente")

gender = st.selectbox("Genero", ["Female", "Male"])
senior_citizen = st.selectbox("Adulto mayor (0=No, 1=Si)", [0, 1])
partner = st.selectbox("Tiene pareja", ["Yes", "No"])
dependents = st.selectbox("Tiene dependientes", ["Yes", "No"])
tenure = st.number_input("Antiguedad en meses", 0, 72, 12)
phone_service = st.selectbox("Servicio telefonico", ["Yes", "No"])
multiple_lines = st.selectbox("Lineas multiples", ["Yes", "No", "No phone service"])
internet_service = st.selectbox("Tipo de internet", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Seguridad online", ["Yes", "No", "No internet service"])
online_backup = st.selectbox("Respaldo online", ["Yes", "No", "No internet service"])
device_protection = st.selectbox("Proteccion de dispositivo", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("Soporte tecnico", ["Yes", "No", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract = st.selectbox("Tipo de contrato", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Facturacion electronica", ["Yes", "No"])
payment_method = st.selectbox("Metodo de pago", [
    "Electronic check", "Mailed check", 
    "Bank transfer (automatic)", "Credit card (automatic)"
])
monthly_charges = st.number_input("Cargo mensual ($)", 0.0, 120.0, 70.0, 0.1)
total_charges = st.number_input("Cargo total acumulado ($)", 0.0, 9000.0, 800.0, 0.1)

# ───────────────────────────────────────────────
# BOTON
# ───────────────────────────────────────────────
st.write("---")
if st.button("PREDECIR CHURN", type="primary"):

    # Feature engineering
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

    # 5 variables derivadas
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

    # Preprocesar
    df = df[input_columns]
    X_proc = preprocessor.transform(df)

    # Predicciones
    resultados = []

    if keras_ok and modelo_keras is not None:
        proba_k = float(modelo_keras.predict(X_proc, verbose=0)[0][0])
        resultados.append(("Keras", proba_k))

    proba_s = float(modelo_sklearn.predict_proba(X_proc)[0][1])
    resultados.append(("Sklearn MLP", proba_s))

    # Mostrar resultados en texto plano (sin metric, sin columns)
    st.write("---")
    st.subheader("RESULTADOS")

    for nombre, proba in resultados:
        st.write(f"**{nombre}:** {proba:.1%} {'-> CHURN' if proba >= 0.5 else '-> NO CHURN'}")

    if len(resultados) == 2:
        promedio = (resultados[0][1] + resultados[1][1]) / 2
        st.write(f"**Promedio:** {promedio:.1%} {'-> CHURN' if promedio >= 0.5 else '-> NO CHURN'}")

        st.write("---")
        st.write("**Interpretacion:**")
        if promedio >= 0.7:
            st.write("ALTO RIESGO: Contactar al cliente inmediatamente.")
        elif promedio >= 0.5:
            st.write("RIESGO MODERADO: Monitorear y ofrecer upgrade.")
        else:
            st.write("BAJO RIESGO: Cliente estable.")
    else:
        st.write("**Interpretacion:**")
        if proba_s >= 0.5:
            st.write("RIESGO DETECTADO: Revisar factores de retencion.")
        else:
            st.write("BAJO RIESGO: Cliente estable.")

    # Factores de riesgo
    st.write("---")
    st.write("**Factores detectados:**")
    if df['is_new_customer'].iloc[0] == 1:
        st.write("- Cliente nuevo")
    if df['electronic_monthly_risk'].iloc[0] == 1:
        st.write("- Pago electronico + mes a mes")
    if df['long_term_contract'].iloc[0] == 0:
        st.write("- Sin contrato largo plazo")
    if df['service_count'].iloc[0] <= 2:
        st.write("- Pocos servicios contratados")
