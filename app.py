# ================================
# APPLICATION IA : CLASSIFICATION DES IRIS (KNN)
# ================================

# Importations
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# ================================
# CHARGEMENT DES DONNÉES
# ================================

iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df['species'] = iris.target
df['species'] = df['species'].map({
    0: 'setosa',
    1: 'versicolor',
    2: 'virginica'
})


# ================================
# PRÉPARATION DES DONNÉES
# ================================

X = df.drop('species', axis=1)
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ================================
# ENTRAÎNEMENT DU MODÈLE
# ================================

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)


# ================================
# INTERFACE STREAMLIT
# ================================

st.set_page_config(
    page_title="Classification des Iris",
    layout="centered"
)

st.title("🌸 Application IA – Classification des Iris")
st.write("Modèle : **K-Nearest Neighbors (KNN)**")
st.write(f"Exactitude du modèle : **{accuracy*100:.2f}%**")

st.markdown("---")

st.subheader("🔢 Entrer les caractéristiques de la fleur")

sepal_length = st.number_input(
    "Longueur du sépale (cm)",
    min_value=0.0,
    value=5.1
)

sepal_width = st.number_input(
    "Largeur du sépale (cm)",
    min_value=0.0,
    value=3.5
)

petal_length = st.number_input(
    "Longueur du pétale (cm)",
    min_value=0.0,
    value=1.4
)

petal_width = st.number_input(
    "Largeur du pétale (cm)",
    min_value=0.0,
    value=0.2
)

if st.button("🔍 Prédire l'espèce"):
    input_data = np.array([
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]).reshape(1, -1)

    input_scaled = scaler.transform(input_data)
    prediction = knn.predict(input_scaled)[0]

    st.success(f"🌼 Espèce prédite : **{prediction.upper()}**")

st.markdown("---")

st.caption(
    "Application développée dans le cadre du TP de classification "
    "des fleurs Iris – Apprentissage automatique."
)


