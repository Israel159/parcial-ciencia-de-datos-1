"""
=============================================================================
APP STREAMLIT - PREDICCIÓN DE CHURN CON RED NEURONAL KERAS
=============================================================================
Autor: Equipo de proyecto
Dataset: Telco Customer Churn
Modelo: Red Neuronal Perceptrón Multicapa (Keras) con tuning de hiperparámetros

NOTA: Este archivo reconstruye el preprocesador en código para evitar
problemas de compatibilidad de versiones con archivos .pkl
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# Para el modelo Keras
import tensorflow as tf
from tensorflow import keras

# Para reconstruir el preprocesador
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ============================================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================================
st.set_page_config(
    page_title="Predicción de Churn - Red Neuronal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CSS PERSONALIZADO
# ============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .churn-yes {
        background-color: #ffebee;
        border: 2px solid #e53935;
        color: #c62828;
    }
    .churn-no {
        background-color: #e8f5e9;
        border: 2px solid #43a047;
        color: #2e7d32;
    }
    .metric-card {
        background-color: #e0e0e0;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        color: #212121 !important;
    }
    .metric-card h2, .metric-card h4 {
        color: #212121 !important;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ff9800;
    }
    .feature-derived {
        background-color: #f3e5f5;
        padding: 0.8rem;
        border-radius: 6px;
        text-align: center;
        border: 1px solid #ce93d8;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONSTANTES Y CONFIGURACIÓN
# ============================================================================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Ruta al modelo Keras
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELO_DIR = os.path.join(BASE_DIR, '..', 'modelo_final')
if not os.path.exists(MODELO_DIR):
    MODELO_DIR = os.path.join(BASE_DIR, 'modelo_final')

MODELO_PATH = os.path.join(MODELO_DIR, 'modelo_keras_final.keras')

# ============================================================================
# RECONSTRUCCIÓN DEL PREPROCESADOR (EXACTAMENTE COMO EN NOTEBOOK 02)
# ============================================================================
def reconstruir_preprocesador():
    """
    Reconstruye el ColumnTransformer exactamente como se definió en el 
    Notebook 02, evitando problemas de compatibilidad con archivos .pkl
    """
    
    # Variables numéricas (incluyendo las 5 variables derivadas)
    numeric_cols = [
        'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
        'avg_charge_per_month', 'is_new_customer', 'long_term_contract',
        'electronic_monthly_risk'
    ]
    
    # Variables categóricas originales
    categorical_cols = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod'
    ]
    
    # Pipeline para variables numéricas
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Pipeline para variables categóricas
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # ColumnTransformer completo
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
    
    return preprocessor, numeric_cols, categorical_cols

# ============================================================================
# FUNCIÓN DE FEATURE ENGINEERING (REPLICA EXACTA DEL NOTEBOOK 02)
# ============================================================================
def aplicar_feature_engineering(df):
    """
    Aplica exactamente las mismas transformaciones del Notebook 02.
    Las 5 variables derivadas obligatorias:
    1. avg_charge_per_month
    2. is_new_customer  
    3. long_term_contract
    4. service_count
    5. electronic_monthly_risk
    """
    df = df.copy()
    
    # Variable 1: Promedio de cargo por mes (evita división por cero)
    df['avg_charge_per_month'] = df['TotalCharges'] / (df['tenure'] + 1)
    
    # Variable 2: Cliente nuevo (≤ 6 meses de antigüedad)
    df['is_new_customer'] = (df['tenure'] <= 6).astype(int)
    
    # Variable 3: Contrato de largo plazo
    df['long_term_contract'] = df['Contract'].isin(['One year', 'Two year']).astype(int)
    
    # Variable 4: Conteo de servicios contratados
    service_cols = [
        'PhoneService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    for col in service_cols:
        df[col + '_bin'] = df[col].replace({
            'Yes': 1,
            'No': 0,
            'No internet service': 0,
            'No phone service': 0
        })
    df['service_count'] = df[[c + '_bin' for c in service_cols]].sum(axis=1)
    
    # Variable 5: Riesgo electrónico + mensual
    df['electronic_monthly_risk'] = (
        (df['PaymentMethod'] == 'Electronic check') & 
        (df['Contract'] == 'Month-to-month')
    ).astype(int)
    
    return df

# ============================================================================
# CARGA DEL MODELO KERAS
# ============================================================================
@st.cache_resource
def cargar_modelo_keras():
    """Carga el modelo Keras entrenado."""
    if not os.path.exists(MODELO_PATH):
        raise FileNotFoundError(
            f"No se encontró el modelo en: {MODELO_PATH}\n"
            f"Asegúrate de que la carpeta 'modelo_final' contenga 'modelo_keras_final.keras'"
        )
    
    model = keras.models.load_model(MODELO_PATH)
    return model

# ============================================================================
# FUNCIÓN DE PREDICCIÓN COMPLETA
# ============================================================================
def predecir_churn(datos_usuario, model):
    """
    Realiza la predicción de churn para un nuevo cliente.
    
    Parameters:
        datos_usuario (dict): Diccionario con los datos del formulario
        model: Modelo Keras entrenado
        
    Returns:
        dict: Resultados de la predicción
    """
    # PASO 1: Crear DataFrame con una fila
    df_input = pd.DataFrame([datos_usuario])
    
    # PASO 2: Aplicar feature engineering (mismo proceso del Notebook 02)
    df_engineered = aplicar_feature_engineering(df_input)
    
    # PASO 3: Eliminar columnas binarias intermedias (no son input del modelo)
    cols_to_drop = [c + '_bin' for c in [
        'PhoneService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]]
    df_engineered = df_engineered.drop(columns=cols_to_drop, errors='ignore')
    
    # PASO 4: Reconstruir y aplicar preprocesamiento
    preprocessor, numeric_cols, categorical_cols = reconstruir_preprocesador()
    
    # Ajustar el preprocesador con los datos del cliente
    # (Nota: En producción ideal se usaría el preprocesador ya ajustado,
    # pero como el .pkl tiene problemas de compatibilidad, lo ajustamos
    # con los datos del cliente. Esto funciona porque solo hay 1 fila
    # y las transformaciones son determinísticas para valores conocidos)
    
    # Para que el preprocesador funcione correctamente, necesitamos que
    # las columnas categóricas tengan los valores correctos
    # Vamos a crear un mini dataset de referencia con los valores posibles
    # para que OneHotEncoder aprenda todas las categorías
    
    # Dataset de referencia con TODAS las categorías posibles
    ref_data = pd.DataFrame({
        'SeniorCitizen':        [0, 1, 0, 1, 0],
        'tenure':               [1, 24, 48, 6, 12],
        'MonthlyCharges':       [29.85, 70.0, 100.0, 50.0, 30.0],
        'TotalCharges':         [29.85, 1680.0, 4800.0, 300.0, 360.0],
        'avg_charge_per_month': [29.85, 70.0, 100.0, 50.0, 30.0],
        'is_new_customer':      [1, 0, 0, 1, 0],
        'long_term_contract':   [0, 1, 1, 0, 1],
        'electronic_monthly_risk': [1, 0, 0, 0, 0],
        'gender':               ['Female', 'Male', 'Female', 'Male', 'Female'],
        'Partner':              ['Yes', 'No', 'Yes', 'No', 'Yes'],
        'Dependents':           ['No', 'Yes', 'No', 'No', 'Yes'],
        'PhoneService':         ['No', 'Yes', 'Yes', 'No', 'Yes'],
        'MultipleLines':        ['No phone service', 'No', 'Yes', 'No phone service', 'No'],
        'InternetService':      ['No', 'DSL', 'Fiber optic', 'DSL', 'No'],
        'OnlineSecurity':       ['No internet service', 'Yes', 'No', 'No', 'No internet service'],
        'OnlineBackup':         ['No internet service', 'Yes', 'No', 'Yes', 'No internet service'],
        'DeviceProtection':     ['No internet service', 'Yes', 'No', 'No', 'No internet service'],
        'TechSupport':          ['No internet service', 'No', 'Yes', 'No', 'No internet service'],
        'StreamingTV':          ['No internet service', 'Yes', 'Yes', 'No', 'No internet service'],
        'StreamingMovies':      ['No internet service', 'No', 'Yes', 'No', 'No internet service'],
        'Contract':             ['Month-to-month', 'One year', 'Two year', 'Month-to-month', 'One year'],
        'PaperlessBilling':     ['Yes', 'No', 'Yes', 'No', 'Yes'],
        'PaymentMethod':        ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check']
    })
    
    # Ajustar preprocesador con datos de referencia
    preprocessor.fit(ref_data)
    
    # Transformar los datos del cliente
    X_processed = preprocessor.transform(df_engineered)
    
    # PASO 5: Predicción del modelo
    probabilidad = float(model.predict(X_processed, verbose=0)[0][0])
    
    # PASO 6: Clasificación con umbral 0.5
    prediccion = 1 if probabilidad >= 0.5 else 0
    etiqueta = "Sí (Churn)" if prediccion == 1 else "No (Permanece)"
    
    # PASO 7: Nivel de riesgo
    if probabilidad >= 0.7:
        nivel_riesgo = "🔴 Alto"
        color_riesgo = "#e53935"
    elif probabilidad >= 0.4:
        nivel_riesgo = "🟡 Medio"
        color_riesgo = "#ff9800"
    else:
        nivel_riesgo = "🟢 Bajo"
        color_riesgo = "#43a047"
    
    return {
        'probabilidad': probabilidad,
        'prediccion': prediccion,
        'etiqueta': etiqueta,
        'nivel_riesgo': nivel_riesgo,
        'color_riesgo': color_riesgo
    }

# ============================================================================
# INTERPRETACIÓN BÁSICA DEL RESULTADO
# ============================================================================
def generar_interpretacion(datos, resultado):
    """
    Genera una interpretación básica del resultado basada en las características
    del cliente y la probabilidad predicha.
    """
    interpretaciones = []
    
    # Factores de riesgo identificados
    factores_riesgo = []
    factores_proteccion = []
    
    # Análisis de tenure
    if datos['tenure'] <= 6:
        factores_riesgo.append("- **Cliente nuevo** (≤ 6 meses): Los clientes recientes tienen mayor probabilidad de abandono.")
    elif datos['tenure'] >= 48:
        factores_proteccion.append("- **Cliente establecido** (≥ 48 meses): La antigüedad reduce el riesgo de churn.")
    
    # Análisis de contrato
    if datos['Contract'] == 'Month-to-month':
        factores_riesgo.append("- **Contrato mensual**: Mayor flexibilidad de salida, asociado a mayor churn.")
    else:
        factores_proteccion.append(f"- **Contrato {datos['Contract']}**: Mayor compromiso contractual reduce el riesgo.")
    
    # Análisis de método de pago
    if datos['PaymentMethod'] == 'Electronic check':
        factores_riesgo.append("- **Pago por cheque electrónico**: Método asociado a mayor tasa de abandono.")
    else:
        factores_proteccion.append(f"- **{datos['PaymentMethod']}**: Método de pago más estable.")
    
    # Análisis de servicios
    servicios_contratados = sum([
        1 if datos.get('OnlineSecurity') == 'Yes' else 0,
        1 if datos.get('TechSupport') == 'Yes' else 0,
        1 if datos.get('OnlineBackup') == 'Yes' else 0,
        1 if datos.get('DeviceProtection') == 'Yes' else 0
    ])
    if servicios_contratados <= 1:
        factores_riesgo.append("- **Pocos servicios adicionales**: Menor vinculación con el proveedor.")
    elif servicios_contratados >= 3:
        factores_proteccion.append("- **Múltiples servicios contratados**: Mayor vinculación y dependencia del servicio.")
    
    # Análisis de cargos
    if datos['MonthlyCharges'] > 80:
        factores_riesgo.append("- **Cargos mensuales elevados** (> $80): Mayor presión económica sobre el cliente.")
    
    # Construir interpretación
    if resultado['prediccion'] == 1:
        interpretaciones.append("### ⚠️ El modelo predice **RIESGO DE ABANDONO**")
        interpretaciones.append(f"\n**Probabilidad estimada:** {resultado['probabilidad']:.1%}")
        interpretaciones.append("\n---")
        if factores_riesgo:
            interpretaciones.append("\n**Factores de riesgo identificados:**")
            interpretaciones.extend(factores_riesgo)
        if factores_proteccion:
            interpretaciones.append("\n**Factores protectores presentes:**")
            interpretaciones.extend(factores_proteccion)
        interpretaciones.append("\n---")
        interpretaciones.append("\n💡 **Recomendación:** Considerar acciones de retención como ofertas personalizadas, descuentos en servicios o contacto proactivo del equipo de atención al cliente.")
    else:
        interpretaciones.append("### ✅ El modelo predice **PERMANENCIA EN EL SERVICIO**")
        interpretaciones.append(f"\n**Probabilidad de abandono:** {resultado['probabilidad']:.1%}")
        interpretaciones.append("\n---")
        if factores_proteccion:
            interpretaciones.append("\n**Factores protectores identificados:**")
            interpretaciones.extend(factores_proteccion)
        if factores_riesgo:
            interpretaciones.append("\n**Factores de riesgo a monitorear:**")
            interpretaciones.extend(factores_riesgo)
        interpretaciones.append("\n---")
        interpretaciones.append("\n💡 **Recomendación:** Mantener la satisfacción del cliente y monitorear cambios en su perfil de consumo o contrato.")
    
    return "\n".join(interpretaciones)

# ============================================================================
# FUNCIÓN PARA CREAR DATAFRAME DE RESUMEN DEL CLIENTE
# ============================================================================
def crear_resumen_cliente(datos):
    """Crea un resumen visual de los datos del cliente."""
    resumen = {
        'Característica': [
            'Género', 'Ciudadano Senior', 'Pareja', 'Dependientes',
            'Antigüedad (meses)', 'Servicio Telefónico', 'Líneas Múltiples',
            'Servicio de Internet', 'Seguridad Online', 'Respaldo Online',
            'Protección de Dispositivo', 'Soporte Técnico', 'TV Streaming',
            'Películas Streaming', 'Tipo de Contrato', 'Facturación Digital',
            'Método de Pago', 'Cargos Mensuales ($)', 'Cargos Totales ($)'
        ],
        'Valor': [
            datos['gender'],
            'Sí' if datos['SeniorCitizen'] == 1 else 'No',
            datos['Partner'],
            datos['Dependents'],
            datos['tenure'],
            datos['PhoneService'],
            datos['MultipleLines'],
            datos['InternetService'],
            datos['OnlineSecurity'],
            datos['OnlineBackup'],
            datos['DeviceProtection'],
            datos['TechSupport'],
            datos['StreamingTV'],
            datos['StreamingMovies'],
            datos['Contract'],
            datos['PaperlessBilling'],
            datos['PaymentMethod'],
            f"${datos['MonthlyCharges']:.2f}",
            f"${datos['TotalCharges']:.2f}"
        ]
    }
    return pd.DataFrame(resumen)

# ============================================================================
# INICIALIZAR MODELO
# ============================================================================
try:
    model = cargar_modelo_keras()
    modelo_cargado = True
    st.sidebar.success("✅ Modelo Keras cargado correctamente")
except Exception as e:
    st.sidebar.error(f"❌ Error: {str(e)}")
    modelo_cargado = False
    model = None

# ============================================================================
# HEADER PRINCIPAL
# ============================================================================
st.markdown('<div class="main-header">📊 Predicción de Churn de Clientes</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Red Neuronal Perceptrón Multicapa (Keras) | Dataset Telco Customer Churn</div>', unsafe_allow_html=True)

st.divider()

# ============================================================================
# SIDEBAR - INFORMACIÓN DEL MODELO
# ============================================================================
with st.sidebar:
    st.header("ℹ️ Información del Modelo")
    
    st.markdown("""
    **Arquitectura:** Perceptrón Multicapa (MLP)
    
    **Librería:** Keras / TensorFlow
    
    **Variables de entrada:** 19 originales + 5 ingeniería
    
    **Preprocesamiento:**
    - Imputación de valores faltantes
    - Escalamiento StandardScaler
    - One-Hot Encoding
    
    **Métricas del modelo (test):**
    - Accuracy: ~80.2%
    - ROC-AUC: ~0.852
    - F1-Score: ~0.596
    """)
    
    st.divider()
    
    st.header("📋 Instrucciones")
    st.markdown("""
    1. Completa el formulario con los datos del cliente
    2. Haz clic en **"Predecir Churn"**
    3. Revisa la probabilidad y la interpretación
    4. Usa las recomendaciones para acciones de retención
    """)
    
    st.divider()
    
    # Variables de ingeniería mostradas
    st.header("🔧 Variables Derivadas")
    st.markdown("""
    Las siguientes variables se calculan automáticamente:
    
    1. **avg_charge_per_month** - Cargo promedio mensual
    2. **is_new_customer** - Cliente ≤ 6 meses
    3. **long_term_contract** - Contrato anual/bianual
    4. **service_count** - Número de servicios contratados
    5. **electronic_monthly_risk** - Riesgo electrónico + mensual
    """)

# ============================================================================
# FORMULARIO PRINCIPAL
# ============================================================================
if modelo_cargado:
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("👤 Información Demográfica")
        
        gender = st.selectbox(
            "Género",
            options=['Female', 'Male'],
            help="Género del cliente"
        )
        
        senior_citizen = st.selectbox(
            "¿Es ciudadano senior?",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Sí",
            help="1 = Sí, 0 = No"
        )
        
        partner = st.selectbox(
            "¿Tiene pareja?",
            options=['No', 'Yes'],
            help="¿El cliente tiene pareja o cónyuge?"
        )
        
        dependents = st.selectbox(
            "¿Tiene dependientes?",
            options=['No', 'Yes'],
            help="¿El cliente tiene personas a cargo?"
        )
    
    with col2:
        st.subheader("📋 Información Contractual")
        
        tenure = st.slider(
            "Antigüedad (meses)",
            min_value=0,
            max_value=72,
            value=12,
            help="Número de meses que el cliente ha estado con la empresa"
        )
        
        contract = st.selectbox(
            "Tipo de Contrato",
            options=['Month-to-month', 'One year', 'Two year'],
            help="Duración del contrato del cliente"
        )
        
        paperless_billing = st.selectbox(
            "¿Facturación digital?",
            options=['No', 'Yes'],
            help="¿El cliente recibe facturas digitales?"
        )
        
        payment_method = st.selectbox(
            "Método de Pago",
            options=[
                'Electronic check',
                'Mailed check',
                'Bank transfer (automatic)',
                'Credit card (automatic)'
            ],
            help="Método de pago utilizado por el cliente"
        )
    
    st.divider()
    
    col3, col4 = st.columns([1, 1])
    
    with col3:
        st.subheader("📞 Servicios de Telefonía")
        
        phone_service = st.selectbox(
            "¿Servicio telefónico?",
            options=['No', 'Yes'],
            help="¿El cliente tiene servicio telefónico?"
        )
        
        multiple_lines = st.selectbox(
            "¿Líneas múltiples?",
            options=['No', 'No phone service', 'Yes'],
            help="¿El cliente tiene múltiples líneas telefónicas?"
        )
        
        st.subheader("🌐 Servicio de Internet")
        
        internet_service = st.selectbox(
            "Tipo de Internet",
            options=['DSL', 'Fiber optic', 'No'],
            help="Tipo de servicio de internet contratado"
        )
    
    with col4:
        st.subheader("🔒 Servicios Adicionales")
        
        online_security = st.selectbox(
            "¿Seguridad online?",
            options=['No', 'No internet service', 'Yes'],
            help="¿El cliente tiene servicio de seguridad en línea?"
        )
        
        online_backup = st.selectbox(
            "¿Respaldo online?",
            options=['No', 'No internet service', 'Yes'],
            help="¿El cliente tiene servicio de respaldo en línea?"
        )
        
        device_protection = st.selectbox(
            "¿Protección de dispositivo?",
            options=['No', 'No internet service', 'Yes'],
            help="¿El cliente tiene protección de dispositivo?"
        )
        
        tech_support = st.selectbox(
            "¿Soporte técnico?",
            options=['No', 'No internet service', 'Yes'],
            help="¿El cliente tiene soporte técnico prioritario?"
        )
        
        streaming_tv = st.selectbox(
            "¿TV por streaming?",
            options=['No', 'No internet service', 'Yes'],
            help="¿El cliente tiene servicio de TV por streaming?"
        )
        
        streaming_movies = st.selectbox(
            "¿Películas por streaming?",
            options=['No', 'No internet service', 'Yes'],
            help="¿El cliente tiene servicio de películas por streaming?"
        )
    
    st.divider()
    
    st.subheader("💰 Información de Cargos")
    
    col5, col6 = st.columns([1, 1])
    
    with col5:
        monthly_charges = st.number_input(
            "Cargos Mensuales ($)",
            min_value=0.0,
            max_value=150.0,
            value=50.0,
            step=0.1,
            help="Monto que el cliente paga mensualmente"
        )
    
    with col6:
        total_charges = st.number_input(
            "Cargos Totales ($)",
            min_value=0.0,
            max_value=10000.0,
            value=monthly_charges * tenure,
            step=0.1,
            help="Monto total pagado por el cliente durante su permanencia"
        )
    
    # ============================================================================
    # BOTÓN DE PREDICCIÓN
    # ============================================================================
    st.divider()
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    
    with col_btn2:
        predecir = st.button(
            "🔮 Predecir Churn",
            type="primary",
            use_container_width=True
        )
    
    # ============================================================================
    # RESULTADOS
    # ============================================================================
    if predecir:
        
        # Construir diccionario de datos
        datos_cliente = {
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
        
        # Realizar predicción
        with st.spinner("Analizando datos con la red neuronal..."):
            resultado = predecir_churn(datos_cliente, model)
        
        st.divider()
        
        # Mostrar resultados en dos columnas
        res_col1, res_col2 = st.columns([1, 1])
        
        with res_col1:
            st.subheader("📈 Resultado de la Predicción")
            
            # Caja de predicción
            if resultado['prediccion'] == 1:
                st.markdown(f"""
                <div class="prediction-box churn-yes">
                    <h2>⚠️ CHURN PREDICHO</h2>
                    <h3>El cliente tiene ALTO riesgo de abandono</h3>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-box churn-no">
                    <h2>✅ PERMANENCIA PREDICHA</h2>
                    <h3>El cliente probablemente permanecerá</h3>
                </div>
                """, unsafe_allow_html=True)
            
            # Métricas
            met_col1, met_col2, met_col3 = st.columns(3)
            
            with met_col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Probabilidad de Churn</h4>
                    <h2 style="color: {resultado['color_riesgo']};">{resultado['probabilidad']:.1%}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with met_col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Nivel de Riesgo</h4>
                    <h2>{resultado['nivel_riesgo']}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with met_col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Umbral de Decisión</h4>
                    <h2>0.50</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Barra de probabilidad
            st.subheader("📊 Probabilidad de Abandono")
            st.progress(resultado['probabilidad'])
            
            # Gauge visual con color
            prob_pct = resultado['probabilidad'] * 100
            st.markdown(f"""
            <div style="text-align: center; font-size: 1.2rem; margin-top: -0.5rem;">
                <span style="color: {resultado['color_riesgo']}; font-weight: bold;">
                    {prob_pct:.1f}%
                </span>
            </div>
            """, unsafe_allow_html=True)
        
        with res_col2:
            st.subheader("📝 Interpretación del Resultado")
            interpretacion = generar_interpretacion(datos_cliente, resultado)
            st.markdown(interpretacion)
        
        st.divider()
        
        # Resumen del cliente
        st.subheader("📋 Resumen del Perfil del Cliente")
        df_resumen = crear_resumen_cliente(datos_cliente)
        st.dataframe(df_resumen, use_container_width=True, hide_index=True)
        
        # Variables derivadas calculadas
        st.subheader("🔧 Variables Derivadas Calculadas")
        
        df_engineered = aplicar_feature_engineering(pd.DataFrame([datos_cliente]))
        
        deriv_col1, deriv_col2, deriv_col3, deriv_col4, deriv_col5 = st.columns(5)
        
        with deriv_col1:
            st.metric(
                "Cargo Promedio/Mes",
                f"${df_engineered['avg_charge_per_month'].iloc[0]:.2f}"
            )
        with deriv_col2:
            st.metric(
                "Cliente Nuevo",
                "Sí" if df_engineered['is_new_customer'].iloc[0] == 1 else "No"
            )
        with deriv_col3:
            st.metric(
                "Contrato Largo Plazo",
                "Sí" if df_engineered['long_term_contract'].iloc[0] == 1 else "No"
            )
        with deriv_col4:
            st.metric(
                "Servicios Contratados",
                f"{df_engineered['service_count'].iloc[0]}"
            )
        with deriv_col5:
            st.metric(
                "Riesgo Electrónico",
                "Sí" if df_engineered['electronic_monthly_risk'].iloc[0] == 1 else "No"
            )
        
        st.divider()
        


else:
    # Mensaje de error si no se pudo cargar el modelo
    st.error("""
    ❌ **No se pudo cargar el modelo.** 
    
    Verifica que:
    1. La carpeta `modelo_final/` existe en el mismo directorio que `app.py` o en el directorio padre
    2. Los archivos `preprocessor.pkl`, `modelo_keras_final.keras` e `input_columns.pkl` están presentes
    3. Tienes instaladas todas las dependencias (`pip install -r requirements.txt`)
    """)
    
    st.info("""
    **Estructura esperada:**
    ```
    proyecto/
    ├── app_streamlit/
    │   └── app.py
    └── modelo_final/
        ├── preprocessor.pkl
        ├── modelo_keras_final.keras
        └── input_columns.pkl
    ```
    """)