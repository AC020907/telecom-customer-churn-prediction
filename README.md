# 📡 Interconnect — Telecom Customer Churn Prediction & Retention System

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-00A65A?style=for-the-badge)](https://lightgbm.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)

Solución de Machine Learning de extremo a extremo desarrollada para el operador de telecomunicaciones **Interconnect**, orientada a predecir la pérdida de clientes (Churn), identificar sus causas raíz y priorizar acciones comerciales de retención en tiempo real.

---

## 💼 Contexto y Problema de Negocio

La retención de clientes es fundamental en la industria de telecomunicaciones, donde adquirir un nuevo suscriptor cuesta entre 5 y 7 veces más que fidelizar a uno existente.

* **Objetivo:** Construir un modelo predictivo capaz de anticipar la cancelación de contratos para activar promociones personalizadas antes de que el cliente abandone el servicio.
* **Reto Principal:** Desbalance significativo de clases (~26.5% de cancelaciones vs. ~73.5% de retención activa) junto a la integración de 4 fuentes relacionales dispares.
* **Impacto del Modelo Final:** El algoritmo seleccionado (**LightGBM**) alcanzó un **AUC-ROC de 0.9453** y una **precisión del 90.99%** en el conjunto de prueba aislado.

---

## 📂 Estructura del Proyecto

```text
telecom-customer-churn-prediction/
│
├── README.md                                 # Presentación técnica y comercial del proyecto
├── requirements.txt                          # Dependencias reproducibles del entorno
├── app.py                                    # Aplicación interactiva en Streamlit
│
├── data/
│   └── raw/                                  # Fuentes de datos primarias (contract, personal, internet, phone)
│
├── notebooks/
│   └── telecom_customer_churn_prediction.ipynb # EDA, ingeniería de variables y modelado
│
└── models/                                   # Pipeline exportado (.pkl)
