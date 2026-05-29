# Introducción

La predicción de abandono de clientes (customer churn) constituye un problema relevante para las organizaciones, debido al impacto económico que representa la pérdida de usuarios y la necesidad de implementar estrategias de retención oportunas.

El presente trabajo tiene como objetivo desarrollar un modelo predictivo basado en redes neuronales profundas tipo perceptrón multicapa (MLP), capaz de estimar la probabilidad de abandono de clientes utilizando datos estructurados.

Para ello, se emplea el dataset Telco Customer Churn, el cual contiene información demográfica, contractual y de consumo de clientes de telecomunicaciones. El desarrollo contempla el ciclo completo de ciencia de datos: análisis exploratorio, preprocesamiento, ingeniería de variables, modelado, tuning, evaluación, interpretabilidad e implementación mediante Streamlit.

# Dataset utilizado

Se empleó el dataset Telco Customer Churn, ampliamente utilizado en problemas de clasificación binaria asociados a retención de clientes.

El conjunto de datos contiene información de 7043 clientes y 21 variables, incluyendo características demográficas, tipo de contrato, servicios contratados y cargos asociados.

La variable objetivo es:

**Churn**
- Yes → cliente abandona el servicio
- No → cliente permanece

El problema corresponde a una tarea de clasificación binaria supervisada.

# Análisis Exploratorio de Datos (EDA)

El análisis exploratorio permitió comprender la estructura del dataset, identificar posibles valores faltantes y analizar la distribución de la variable objetivo.

Se observó que la mayoría de variables son categóricas, relacionadas con servicios contratados, tipo de pago y condiciones contractuales.

Respecto a valores faltantes, se identificaron registros vacíos principalmente en la variable TotalCharges, situación que será abordada durante el preprocesamiento.

Asimismo, la distribución del churn evidencia una mayor proporción de clientes que permanecen respecto a aquellos que abandonan el servicio, lo que sugiere un moderado desbalance de clases.