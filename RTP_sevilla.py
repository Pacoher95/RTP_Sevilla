#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  2 16:11:53 2025

@author: pacohermosilla
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA

st.set_page_config(page_title="Retorno al juego - IA vs Humanos", layout="wide")
st.title("🏥 Evaluación del Retorno al Juego: ¿Está listo este jugador?")

# Ruta para guardar respuestas
csv_path = "registro_retorno.csv"

# Simular base de datos
np.random.seed(42)
n = 500

data = pd.DataFrame({
    'Fuerza_CMJ': np.random.normal(40, 10, n),
    'Tiempo_TTest': np.random.normal(10.5, 1.2, n),
    'Asimetria': np.random.normal(6, 3, n),
    'Dolor_reportado': np.random.randint(0, 2, n),
    'Confianza_auto': np.random.normal(7, 2, n),
    'Alta_medica': np.random.randint(0, 2, n),
})

data['Apto_retorno'] = (
    (data['Fuerza_CMJ'] > 28) &
    (data['Tiempo_TTest'] < 11) &
    (data['Asimetria'] < 10) &
    (data['Dolor_reportado'] == 0) &
    (data['Confianza_auto'] > 4) &
    (data['Alta_medica'] == 1)
).astype(int)

# Entrenar modelo
X = data.drop('Apto_retorno', axis=1)
y = data['Apto_retorno']
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Entrenar modelo KNN para estimar confianza local
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X, y)

st.markdown("---")
st.subheader("🧍 Interactúa con el jugador lesionado")

cols = st.columns(3)
with cols[0]:
    cmj = st.slider("🦵 Altura CMJ (cm)", 20.0, 60.0, 40.0)
    ttest = st.slider("🏃‍♂️ Tiempo T-Test (s)", 8.0, 15.0, 10.5)
with cols[1]:
    asimetria = st.slider("↔️ Asimetría pierna dominante (%)", 0.0, 15.0, 6.0)
    confianza = st.slider("🧠 Confianza del jugador (0-10)", 0, 10, 7)
with cols[2]:
    dolor = st.radio("❌ ¿Reporta dolor?", ["Sí", "No"])
    dolor_val = 1 if dolor == "Sí" else 0
    alta_medica = st.radio("📋 ¿Tiene el alta médica?", ["Sí", "No"])
    alta_val = 1 if alta_medica == "Sí" else 0

entrada = pd.DataFrame({
    'Fuerza_CMJ': [cmj],
    'Tiempo_TTest': [ttest],
    'Asimetria': [asimetria],
    'Dolor_reportado': [dolor_val],
    'Confianza_auto': [confianza],
    'Alta_medica': [alta_val]
})

# Evaluación del usuario
st.markdown("**¿Tú dejarías jugar a este jugador?**")
decision_usuario = st.radio("Tu decisión:", ["✅ Sí, puede volver a jugar", "⛔ No, aún no está listo"])
decision_val = 1 if decision_usuario.startswith("✅") else 0

# Evaluación del modelo
pred = model.predict(entrada)[0]
proba = model.predict_proba(entrada)[0][1]

# Calcular precisión esperada (confianza KNN local)
knn_preds = knn.predict(X)
knn_accuracy = np.mean(knn_preds == y)
local_preds = knn.kneighbors(entrada, return_distance=False)
local_labels = y.iloc[local_preds[0]]
confianza_local = local_labels.value_counts(normalize=True).max()

resultado_mostrado = False

# Guardar respuesta y mostrar resultado después
if st.button("📩 Registrar mi evaluación"):
    registro = entrada.copy()
    registro['Decision_usuario'] = decision_val
    registro['Decision_modelo'] = pred
    registro['Coincide'] = int(decision_val == pred)

    if os.path.exists(csv_path):
        anterior = pd.read_csv(csv_path)
        actualizado = pd.concat([anterior, registro], ignore_index=True)
    else:
        actualizado = registro

    actualizado.to_csv(csv_path, index=False)
    st.success("Tu evaluación ha sido registrada correctamente.")
    resultado_mostrado = True

if resultado_mostrado:
    st.markdown("---")
    col4, col5 = st.columns(2)
    with col4:
        st.subheader("🧠 Resultado del modelo")
        if pred == 1:
            st.success(f"✅ El modelo indica que puede volver a jugar. Confianza: {proba:.2f}")
        else:
            st.error(f"⛔ El modelo indica que NO debe volver. Confianza: {1 - proba:.2f}")

        st.info(f"🔍 Precisión estimada del modelo para entradas similares: {confianza_local * 100:.1f}%")

        # Visualizar la posición del jugador vs vecinos
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        entrada_pca = pca.transform(entrada)
        X_knn_pca = pca.transform(X.iloc[local_preds[0]])

        fig, ax = plt.subplots()
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette={0: 'skyblue', 1: 'mediumseagreen'}, alpha=0.6, ax=ax)
        ax.scatter(entrada_pca[0, 0], entrada_pca[0, 1], color='red', s=120, label='Jugador actual')
        ax.scatter(X_knn_pca[:, 0], X_knn_pca[:, 1], edgecolors='black', facecolors='none', s=100, label='Vecinos cercanos')
        ax.set_title("Ubicación del jugador respecto a casos conocidos")
        ax.legend()
        st.pyplot(fig)

    with col5:
        if decision_val == pred:
            st.info("🎯 Tu criterio coincide con el modelo.")
        else:
            st.warning("⚠️ Tu decisión difiere de la del modelo.")

# Mostrar resultados
if os.path.exists(csv_path):
    st.markdown("---")
    st.subheader("📈 Resultados en tiempo real de los asistentes")
    data_total = pd.read_csv(csv_path)

    col6, col7 = st.columns(2)
    col6.metric("👥 Participantes", len(data_total))
    aciertos = data_total['Coincide'].sum()
    porcentaje = (aciertos / len(data_total)) * 100
    col7.metric("🎯 Coincidencia con el modelo", f"{porcentaje:.1f}%")

    fig, ax = plt.subplots()
    sns.countplot(data=data_total, x='Coincide', palette='coolwarm', ax=ax)
    ax.set_xticklabels(['Difiere del modelo', 'Coincide con el modelo'])
    ax.set_ylabel("Número de personas")
    ax.set_title("Comparación entre criterio humano y modelo")
    st.pyplot(fig)

    with st.expander("📝 Ver respuestas recientes"):
        st.dataframe(data_total.tail(10))

    csv = data_total.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Descargar resultados en CSV", csv, "evaluaciones_retorno.csv", "text/csv")

st.caption("Esta aplicación demuestra cómo los modelos de IA pueden apoyar decisiones clínicas en el retorno deportivo.")
