# ================================
# APPLICATION IA : CLASSIFICATION DES IRIS (KNN)
# ================================

# Importations
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

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
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS personnalisés pour le thème sombre
st.markdown("""
    <style>
        /* Thème sombre personnalisé */
        .main {
            background-color: #0e1117;
        }
        
        .stMetric {
            background-color: #161b22;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #00d4ff;
        }
        
        .stButton > button {
            background-color: #00d4ff;
            color: #0e1117;
            font-weight: bold;
            border-radius: 8px;
            padding: 10px 30px;
        }
        
        .stButton > button:hover {
            background-color: #00b8cc;
            box-shadow: 0 0 15px rgba(0, 212, 255, 0.5);
        }
        
        h1, h2, h3, h4, h5, h6 {
            color: #c9d1d9;
            font-weight: bold;
        }
        
        .stMarkdown {
            color: #c9d1d9;
        }
        
        .stNumberInput label {
            color: #c9d1d9;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🌸 Application IA – Classification des Iris")

# Affichage des métriques en haut
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("📊 Modèle utilisé", "K-Nearest Neighbors")

with col2:
    st.metric("🎯 Exactitude", f"{accuracy*100:.2f}%")

with col3:
    st.metric("📈 Voisins (K)", "3")

st.markdown("---")

st.subheader("🔢 Prédiction - Entrer les caractéristiques de la fleur")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input(
        "Longueur du sépale (cm)",
        min_value=0.0,
        value=5.1
    )
    
    petal_length = st.number_input(
        "Longueur du pétale (cm)",
        min_value=0.0,
        value=1.4
    )

with col2:
    sepal_width = st.number_input(
        "Largeur du sépale (cm)",
        min_value=0.0,
        value=3.5
    )
    
    petal_width = st.number_input(
        "Largeur du pétale (cm)",
        min_value=0.0,
        value=0.2
    )

if st.button("🔍 Prédire l'espèce", use_container_width=True):
    input_data = np.array([
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]).reshape(1, -1)

    input_scaled = scaler.transform(input_data)
    prediction = knn.predict(input_scaled)[0]
    
    # Affichage du résultat avec icônes selon l'espèce
    emoji_dict = {
        "setosa": "🌹",
        "versicolor": "🌺",
        "virginica": "🌻"
    }
    
    emoji = emoji_dict.get(prediction, "🌸")
    
    st.success(f"{emoji} Espèce prédite : **{prediction.upper()}**", icon="✅")

st.markdown("---")

st.info("📚 **Section Analyses et Visualisations** - Explorez les données détaillées ci-dessous", icon="ℹ️")

# ================================
# VISUALISATIONS ET ANALYSES
# ================================

st.markdown("---")
st.header("📊 Analyses et Visualisations")

# Exercice 1 : Effectifs des espèces
st.subheader("Exercice 1 : Effectifs des espèces")

effectifs = df['species'].value_counts()

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Distribution des espèces**")
    fig, ax = plt.subplots(figsize=(8, 5))
    effectifs.plot(kind="bar", color=["green", "orange", "blue"], ax=ax)
    ax.set_title("Histogramme des effectifs des espèces")
    ax.set_xlabel("Espèces")
    ax.set_ylabel("Effectif")
    ax.grid(axis="y")
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    st.markdown("**Répartition en secteurs**")
    fig, ax = plt.subplots(figsize=(8, 5))
    effectifs.plot(kind="pie", autopct="%1.1f%%", ax=ax)
    ax.set_title("Répartition des espèces")
    ax.set_ylabel("")
    plt.tight_layout()
    st.pyplot(fig)

# Exercice 2 : Variables quantitatives
st.subheader("Exercice 2 : Analyse des variables quantitatives")

st.markdown("**Résumé statistique de la longueur du pétale**")
st.write(df["petal length (cm)"].describe())

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df["petal length (cm)"], bins=20, edgecolor="black", color="skyblue")
ax.set_title("Histogramme de la longueur des pétales")
ax.set_xlabel("Longueur du pétale (cm)")
ax.set_ylabel("Effectif")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
st.pyplot(fig)

st.markdown("**Histogrammes des autres variables quantitatives**")

variables = [
    "petal width (cm)",
    "sepal length (cm)",
    "sepal width (cm)"
]

cols = st.columns(3)
for idx, var in enumerate(variables):
    with cols[idx]:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(df[var], bins=20, edgecolor="black", color="lightcoral")
        ax.set_title(f"Histogramme de {var}")
        ax.set_xlabel(var)
        ax.set_ylabel("Effectif")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

# Exercice 3 : Étude bivariée
st.subheader("Exercice 3 : Étude bivariée (corrélations entre variables)")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Pétale : Longueur vs Largeur**")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["petal length (cm)"], df["petal width (cm)"], alpha=0.6, s=100)
    ax.set_title("Relation : longueur vs largeur du pétale")
    ax.set_xlabel("Longueur du pétale (cm)")
    ax.set_ylabel("Largeur du pétale (cm)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    st.markdown("**Sépale : Longueur vs Largeur**")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["sepal length (cm)"], df["sepal width (cm)"], alpha=0.6, s=100, color="green")
    ax.set_title("Relation : longueur vs largeur du sépale")
    ax.set_xlabel("Longueur du sépale (cm)")
    ax.set_ylabel("Largeur du sépale (cm)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

# Exercice 4 : Boxplots par espèce
st.subheader("Exercice 4 : Analyse par espèce (Boxplots)")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Longueur du pétale selon l'espèce**")
    fig, ax = plt.subplots(figsize=(8, 5))
    df.boxplot(column="petal length (cm)", by="species", ax=ax)
    ax.set_title("Longueur du pétale selon l'espèce")
    ax.set_xlabel("Espèce")
    ax.set_ylabel("Longueur du pétale (cm)")
    plt.suptitle("")
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    st.markdown("**Largeur du pétale selon l'espèce**")
    fig, ax = plt.subplots(figsize=(8, 5))
    df.boxplot(column="petal width (cm)", by="species", ax=ax)
    ax.set_title("Largeur du pétale selon l'espèce")
    ax.set_xlabel("Espèce")
    ax.set_ylabel("Largeur du pétale (cm)")
    plt.suptitle("")
    plt.tight_layout()
    st.pyplot(fig)

# Exercice 5 : Intégration de l'espèce avec matrice de corrélation
st.subheader("Exercice 5 : Nuage de points par espèce et corrélation")

st.markdown("**Matrice de corrélation des variables quantitatives**")
correlation = df.drop("species", axis=1).corr()
st.write(correlation)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap="coolwarm", center=0, ax=ax, square=True)
ax.set_title("Matrice de corrélation des variables Iris")
plt.tight_layout()
st.pyplot(fig)

st.markdown("**Nuage de points coloré par espèce**")
fig, ax = plt.subplots(figsize=(10, 6))

colors = {"setosa": "red", "versicolor": "blue", "virginica": "green"}

for esp in df["species"].unique():
    sous_df = df[df["species"] == esp]
    ax.scatter(
        sous_df["petal length (cm)"],
        sous_df["petal width (cm)"],
        label=esp,
        alpha=0.7,
        s=100,
        color=colors[esp]
    )

ax.set_title("Nuage de points : longueur vs largeur du pétale par espèce")
ax.set_xlabel("Longueur du pétale (cm)")
ax.set_ylabel("Largeur du pétale (cm)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
st.pyplot(fig)

# ================================
# PIED DE PAGE
# ================================

st.markdown("---")
st.caption(
    "🚀 Application développée dans le cadre du TP de classification "
    "des fleurs Iris – Apprentissage automatique."
)


