import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras

# ───────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ───────────────────────────────────────────────
st.set_page_config(
    page_title="Predicción de Churn - Telco",
    page_icon="📊",
    layout="centered"
)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ───────────────────────────────────────────────
# CARGAR MODELOS Y PREPROCESAMIENTO
# ───────────────────────────────────────────────
@st.cache_resource
def cargar_artifacts():
    """Carga el preprocessor, columnas de entrada y modelos entrenados."""
    # Ajusta estas rutas según tu estructura de carpetas
    preprocessor = joblib.load("modelo_final/preprocessor.pkl")
    input_columns = joblib.load("modelo_final/input_columns.pkl")

    modelo_keras = keras.models.load_model("modelo_final/modelo_keras_final.keras")
    modelo_sklearn = joblib.load("modelo_final/modelo_sklearn_final.pkl")

    return preprocessor, input_columns, modelo_keras, modelo_sklearn

try:
    preprocessor, input_columns, modelo_keras, modelo_sklearn = cargar_artifacts()
    modelos_cargados = True
except Exception as e:
    st.error(f"❌ Error al cargar modelos: {e}")
    st.info("Asegúrate de que la carpeta `modelo_final/` esté en la misma ubicación que esta app.")
    modelos_cargados = False

# ───────────────────────────────────────────────
# TÍTULO Y DESCRIPCIÓN
# ───────────────────────────────────────────────
st.title("📡 Predicción de Abandono de Clientes (Churn)")
st.markdown("""
Esta aplicación predice la probabilidad de que un cliente abandone el servicio de telecomunicaciones,
utilizando una red neuronal entrenada con el dataset **Telco Customer Churn**.

**Desarrollado para:** Examen Parcial 2026-1 — Redes Neuronales Perceptrón Multicapa
""")

st.divider()

# ───────────────────────────────────────────────
# FORMULARIO DE ENTRADA
# ───────────────────────────────────────────────
st.subheader("📝 Ingrese los datos del cliente")

with st.form("formulario_cliente"):
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Género", ["Female", "Male"])
        senior_citizen = st.selectbox("¿Adulto mayor?", [0, 1], format_func=lambda x: "Sí" if x==1 else "No")
        partner = st.selectbox("¿Tiene pareja?", ["Yes", "No"])
        dependents = st.selectbox("¿Tiene dependientes?", ["Yes", "No"])
        tenure = st.number_input("Antigüedad (meses)", min_value=0, max_value=72, value=12)
        phone_service = st.selectbox("¿Servicio telefónico?", ["Yes", "No"])
        multiple_lines = st.selectbox("¿Líneas múltiples?", ["Yes", "No", "No phone service"])

    with col2:
        internet_service = st.selectbox("Internet", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Seguridad online", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Respaldo online", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Protección de dispositivo", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Soporte técnico", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    with col3:
        contract = st.selectbox("Contrato", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("¿Facturación electrónica?", ["Yes", "No"])
        payment_method = st.selectbox("Método de pago", [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ])
        monthly_charges = st.number_input("Cargo mensual ($)", min_value=0.0, max_value=120.0, value=70.0, step=0.1)
        total_charges = st.number_input("Cargo total acumulado ($)", min_value=0.0, max_value=9000.0, value=800.0, step=0.1)

    submitted = st.form_submit_button("🔮 Predecir Churn", use_container_width=True)

# ───────────────────────────────────────────────
# FUNCIÓN DE FEATURE ENGINEERING
# ───────────────────────────────────────────────
def aplicar_feature_engineering(df):
    """Replica exactamente las 5 variables derivadas del Notebook 02."""
    df = df.copy()

    # 1. Cargo promedio por mes
    df['avg_charge_per_month'] = df['TotalCharges'] / (df['tenure'] + 1)

    # 2. Cliente nuevo (≤ 6 meses)
    df['is_new_customer'] = (df['tenure'] <= 6).astype(int)

    # 3. Contrato a largo plazo
    df['long_term_contract'] = df['Contract'].isin(['One year', 'Two year']).astype(int)

    # 4. Cantidad de servicios contratados
    service_cols = [
        'PhoneService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    for col in service_cols:
        df[f"{col}_bin"] = df[col].replace({
            'Yes': 1,
            'No': 0,
            'No internet service': 0,
            'No phone service': 0
        })
    bin_cols = [f"{c}_bin" for c in service_cols]
    df['service_count'] = df[bin_cols].sum(axis=1)

    # 5. Riesgo: pago electrónico + contrato mes a mes
    df['electronic_monthly_risk'] = (
        (df['PaymentMethod'] == 'Electronic check') &
        (df['Contract'] == 'Month-to-month')
    ).astype(int)

    return df

# ───────────────────────────────────────────────
# PREDICCIÓN
# ───────────────────────────────────────────────
if submitted and modelos_cargados:
    # 1. Construir DataFrame original
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

    # 2. Feature Engineering
    df_engineered = aplicar_feature_engineering(df_input)

    # 3. Reordenar columnas exactamente como espera el preprocessor
    df_engineered = df_engineered[input_columns]

    # 4. Preprocesar (imputar, escalar, one-hot)
    X_processed = preprocessor.transform(df_engineered)

    # 5. Predicciones
    proba_keras = float(modelo_keras.predict(X_processed, verbose=0)[0][0])
    proba_sklearn = float(modelo_sklearn.predict_proba(X_processed)[0][1])

    # 6. Resultados
    st.divider()
    st.subheader("📊 Resultados de la Predicción")

    col_k, col_s, col_int = st.columns(3)

    with col_k:
        st.metric(
            label="🔴 Keras (Modelo Final)",
            value=f"{proba_keras:.1%}",
            delta="Churn" if proba_keras >= 0.5 else "No Churn",
            delta_color="inverse" if proba_keras >= 0.5 else "normal"
        )

    with col_s:
        st.metric(
            label="🟢 Scikit-learn (MLP)",
            value=f"{proba_sklearn:.1%}",
            delta="Churn" if proba_sklearn >= 0.5 else "No Churn",
            delta_color="inverse" if proba_sklearn >= 0.5 else "normal"
        )

    with col_int:
        proba_promedio = (proba_keras + proba_sklearn) / 2
        st.metric(
            label="⚖️ Promedio Ensemble",
            value=f"{proba_promedio:.1%}",
            delta="Churn" if proba_promedio >= 0.5 else "No Churn",
            delta_color="inverse" if proba_promedio >= 0.5 else "normal"
        )

    # 7. Barra de progreso visual
    st.divider()
    st.write("**Nivel de riesgo visual (promedio):**")
    st.progress(min(proba_promedio, 1.0))

    # 8. Interpretación básica
    st.subheader("💡 Interpretación del Resultado")

    if proba_promedio >= 0.7:
        st.error("""
        **🔴 ALTO RIESGO DE CHURN**
        El cliente presenta una alta probabilidad de abandono. Se recomienda:
        - Contactar al cliente de forma inmediata.
        - Ofrecer descuentos o beneficios de retención.
        - Revisar si tiene contrato mes-a-mes + pago electrónico (factor de riesgo clave).
        """)
    elif proba_promedio >= 0.5:
        st.warning("""
        **🟡 RIESGO MODERADO DE CHURN**
        El cliente está en zona de riesgo. Acciones sugeridas:
        - Monitorear su comportamiento de uso.
        - Evaluar ofertas de upgrade o contratos a mayor plazo.
        - Verificar satisfacción con servicios contratados.
        """)
    else:
        st.success("""
        **🟢 BAJO RIESGO DE CHURN**
        El cliente es probable que permanezca. Estrategia:
        - Mantener servicio estándar.
        - Ofrecer cross-selling de nuevos servicios.
        - Continuar con programas de fidelización.
        """)

    # 9. Factores de riesgo identificados
    st.subheader("📋 Factores de Riesgo del Cliente Ingresado")

    factores = []
    if df_engineered['is_new_customer'].iloc[0] == 1:
        factores.append("• Cliente nuevo (≤ 6 meses de antigüedad)")
    if df_engineered['electronic_monthly_risk'].iloc[0] == 1:
        factores.append("• Pago con cheque electrónico + contrato mes a mes")
    if df_engineered['long_term_contract'].iloc[0] == 0:
        factores.append("• Contrato a corto plazo (sin compromiso)")
    if df_engineered['service_count'].iloc[0] <= 2:
        factores.append("• Baja cantidad de servicios contratados")
    if df_engineered['avg_charge_per_month'].iloc[0] > df_engineered['MonthlyCharges'].iloc[0]:
        factores.append("• Cargo promedio histórico mayor al cargo mensual actual")

    if factores:
        for f in factores:
            st.write(f)
    else:
        st.write("• No se detectaron factores de riesgo adicionales destacados.")

    # 10. Datos técnicos (expandible)
    with st.expander("🔧 Ver datos técnicos del modelo"):
        st.write("**Forma del vector de entrada procesado:**", X_processed.shape)
        st.write("**Columnas originales + ingeniería:**", list(input_columns))
        st.write("**Probabilidad Keras (raw):**", f"{proba_keras:.6f}")
        st.write("**Probabilidad Sklearn (raw):**", f"{proba_sklearn:.6f}")

elif submitted and not modelos_cargados:
    st.error("No se pueden realizar predicciones porque los modelos no se cargaron correctamente.")
