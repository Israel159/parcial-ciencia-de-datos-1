import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras
from pathlib import Path

# ───────────────────────────────────────────────
# CONFIGURACIÓN
# ───────────────────────────────────────────────
st.set_page_config(
    page_title="Prediccion de Churn - Telco",
    page_icon=":bar_chart:",
    layout="centered"
)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ───────────────────────────────────────────────
# RUTAS
# ───────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "modelo_final"

# ───────────────────────────────────────────────
# CARGAR ARTEFACTOS
# ───────────────────────────────────────────────
@st.cache_resource
def cargar_artifacts():
    preprocessor = joblib.load(MODEL_DIR / "preprocessor.pkl")
    input_columns = joblib.load(MODEL_DIR / "input_columns.pkl")
    modelo_keras = keras.models.load_model(MODEL_DIR / "modelo_keras_final.keras")
    modelo_sklearn = joblib.load(MODEL_DIR / "modelo_sklearn_final.pkl")
    return preprocessor, input_columns, modelo_keras, modelo_sklearn

try:
    preprocessor, input_columns, modelo_keras, modelo_sklearn = cargar_artifacts()
    modelos_cargados = True
except Exception as e:
    st.error("Error al cargar modelos: " + str(e))
    st.info("Verifica que la carpeta modelo_final/ este al mismo nivel que app_streamlit/")
    modelos_cargados = False

# ───────────────────────────────────────────────
# TITULO
# ───────────────────────────────────────────────
st.title("Prediccion de Abandono de Clientes (Churn)")
st.markdown("App desarrollada para el Examen Parcial 2026-1 - Redes Neuronales Perceptron Multicapa")
st.markdown("Dataset: **Telco Customer Churn**")
st.divider()

# ───────────────────────────────────────────────
# ENTRADAS DEL USUARIO (sin st.form, mas estable)
# ───────────────────────────────────────────────
st.subheader("Ingrese los datos del cliente")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Genero", ["Female", "Male"])
    senior_citizen = st.selectbox("Adulto mayor", [0, 1], format_func=lambda x: "Si" if x==1 else "No")
    partner = st.selectbox("Tiene pareja", ["Yes", "No"])
    dependents = st.selectbox("Tiene dependientes", ["Yes", "No"])
    tenure = st.number_input("Antiguedad (meses)", min_value=0, max_value=72, value=12)
    phone_service = st.selectbox("Servicio telefonico", ["Yes", "No"])
    multiple_lines = st.selectbox("Lineas multiples", ["Yes", "No", "No phone service"])

with col2:
    internet_service = st.selectbox("Internet", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Seguridad online", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Respaldo online", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Proteccion dispositivo", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Soporte tecnico", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

with col3:
    contract = st.selectbox("Contrato", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Facturacion electronica", ["Yes", "No"])
    payment_method = st.selectbox("Metodo de pago", [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ])
    monthly_charges = st.number_input("Cargo mensual ($)", min_value=0.0, max_value=120.0, value=70.0, step=0.1)
    total_charges = st.number_input("Cargo total ($)", min_value=0.0, max_value=9000.0, value=800.0, step=0.1)

# ───────────────────────────────────────────────
# BOTON DE PREDICCION (fuera de form, mas estable)
# ───────────────────────────────────────────────
st.divider()
predict_btn = st.button("Predecir Churn", use_container_width=True, type="primary")

# ───────────────────────────────────────────────
# FEATURE ENGINEERING
# ───────────────────────────────────────────────
def aplicar_feature_engineering(df):
    df = df.copy()
    df['avg_charge_per_month'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['is_new_customer'] = (df['tenure'] <= 6).astype(int)
    df['long_term_contract'] = df['Contract'].isin(['One year', 'Two year']).astype(int)

    service_cols = [
        'PhoneService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    for col in service_cols:
        df[col + '_bin'] = df[col].replace({
            'Yes': 1, 'No': 0,
            'No internet service': 0, 'No phone service': 0
        })
    bin_cols = [c + '_bin' for c in service_cols]
    df['service_count'] = df[bin_cols].sum(axis=1)

    df['electronic_monthly_risk'] = (
        (df['PaymentMethod'] == 'Electronic check') &
        (df['Contract'] == 'Month-to-month')
    ).astype(int)
    return df

# ───────────────────────────────────────────────
# PREDICCION Y RESULTADOS
# ───────────────────────────────────────────────
if predict_btn and modelos_cargados:
    datos = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }

    df_input = pd.DataFrame([datos])
    df_engineered = aplicar_feature_engineering(df_input)
    df_engineered = df_engineered[input_columns]
    X_processed = preprocessor.transform(df_engineered)

    proba_keras = float(modelo_keras.predict(X_processed, verbose=0)[0][0])
    proba_sklearn = float(modelo_sklearn.predict_proba(X_processed)[0][1])
    proba_promedio = (proba_keras + proba_sklearn) / 2

    st.divider()
    st.subheader("Resultados de la Prediccion")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Keras (Final)", f"{proba_keras:.1%}")
        if proba_keras >= 0.5:
            st.error("Churn")
        else:
            st.success("No Churn")
    with c2:
        st.metric("Sklearn (MLP)", f"{proba_sklearn:.1%}")
        if proba_sklearn >= 0.5:
            st.error("Churn")
        else:
            st.success("No Churn")
    with c3:
        st.metric("Promedio", f"{proba_promedio:.1%}")
        if proba_promedio >= 0.5:
            st.error("Churn")
        else:
            st.success("No Churn")

    st.divider()
    st.write("Nivel de riesgo:")
    st.progress(min(proba_promedio, 1.0))

    st.subheader("Interpretacion")
    if proba_promedio >= 0.7:
        st.error("ALTO RIESGO DE CHURN. Se recomienda contactar al cliente de inmediato y ofrecer beneficios de retencion.")
    elif proba_promedio >= 0.5:
        st.warning("RIESGO MODERADO. Monitorear comportamiento y evaluar ofertas de upgrade.")
    else:
        st.success("BAJO RIESGO. Cliente probable que permanezca. Estrategia de fidelizacion estandar.")

    st.subheader("Factores de riesgo detectados")
    factores = []
    if df_engineered['is_new_customer'].iloc[0] == 1:
        factores.append("- Cliente nuevo (6 meses o menos)")
    if df_engineered['electronic_monthly_risk'].iloc[0] == 1:
        factores.append("- Pago con cheque electronico + contrato mes a mes")
    if df_engineered['long_term_contract'].iloc[0] == 0:
        factores.append("- Contrato a corto plazo")
    if df_engineered['service_count'].iloc[0] <= 2:
        factores.append("- Baja cantidad de servicios contratados")
    if df_engineered['avg_charge_per_month'].iloc[0] > df_engineered['MonthlyCharges'].iloc[0]:
        factores.append("- Cargo promedio historico mayor al actual")

    if factores:
        for f in factores:
            st.write(f)
    else:
        st.write("- No se detectaron factores de riesgo adicionales.")

    with st.expander("Ver datos tecnicos"):
        st.write("Vector procesado shape:", X_processed.shape)
        st.write("Probabilidad Keras:", f"{proba_keras:.6f}")
        st.write("Probabilidad Sklearn:", f"{proba_sklearn:.6f}")

elif predict_btn and not modelos_cargados:
    st.error("No se pueden realizar predicciones. Los modelos no se cargaron.")
