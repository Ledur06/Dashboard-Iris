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
# CONFIGURATION STREAMLIT
# ================================

st.set_page_config(
    page_title="Classification des Iris",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================
# THÈME VISUEL NOIR & BLANC PRO
# ================================

st.markdown("""
<style>

/* -------- BASE -------- */
html, body, [class*="css"] {
    font-family: "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

/* Fond principal */
.stApp {
    background-color: #f4f4f4;
}

/* -------- TEXTE -------- */
h1, h2, h3, h4, h5, h6 {
    color: #111111;
    font-weight: 600;
}

p, span, label, div {
    color: #222222;
    font-size: 15px;
}

/* -------- CARTES -------- */
.card {
    background-color: #ffffff;
    border-radius: 14px;
    padding: 24px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.08);
    margin-bottom: 24px;
}

/* -------- METRICS -------- */
[data-testid="metric-container"] {
    background-color: #ffffff;
    border-radius: 14px;
    padding: 18px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    border-left: 6px solid #111111;
}

[data-testid="metric-container"] * {
    color: #111111 !important;
}

/* -------- INPUTS -------- */
input, select {
    background-color: #ffffff !important;
    color: #111111 !important;
    border-radius: 8px !important;
}

/* -------- BOUTONS -------- */
.stButton > button {
    background-color: #111111;
    color: #ffffff;
    border-radius: 10px;
    padding: 12px 28px;
    font-weight: 600;
    border: none;
    transition: all 0.25s ease;
}

.stButton > button:hover {
    background-color: #000000;
    transform: translateY(-1px);
}

/* -------- SIDEBAR -------- */
section[data-testid="stSidebar"] {
    background-color: #111111;
}

section[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

/* -------- ALERTES -------- */
.stSuccess, .stInfo {
    background-color: #ffffff;
    color: #111111;
    border-radius: 12px;
    padding: 18px;
    border-left: 6px solid #111111;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
}

/* -------- SEPARATEURS -------- */
hr {
    border: none;
    height: 1px;
    background-color: #dddddd;
    margin: 32px 0;
}

</style>
""", unsafe_allow_html=True)


# ================================
# MENU DE NAVIGATION (SIDEBAR)
# ================================

st.sidebar.title("🌸 Navigation")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Sélectionnez une page :",
    ["🏠 Accueil", "🔮 Prédiction", "📊 Analyses", "📈 Détails Modèle"]
)

st.sidebar.markdown("---")
st.sidebar.info("💡 Utilisez le menu ci-dessus pour naviguer dans l'application")


# ================================
# PAGE 1 : ACCUEIL
# ================================

if page == "🏠 Accueil":
    st.title("🌸 Application IA – Classification des Iris")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📊 Modèle utilisé", "K-Nearest Neighbors")
    
    with col2:
        st.metric("🎯 Exactitude", f"{accuracy*100:.2f}%")
    
    with col3:
        st.metric("📈 Voisins (K)", "3")
    
    st.markdown("---")
    
    st.markdown("""
    <div style="background-color: #ffffff; border-radius: 12px; padding: 25px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);">
        <h3 style="color: #1a1a1a; margin-top: 0;">📚 À propos de cette application</h3>
        <p style="color: #333333; line-height: 1.6;">
            <b>Objectif :</b> Classifier les fleurs Iris en trois espèces : Setosa, Versicolor et Virginica.<br><br>
            <b>Modèle :</b> L'algorithme K-Nearest Neighbors (KNN) analyse les caractéristiques (longueur et largeur des sépales et pétales) 
            pour prédire l'espèce de la fleur.<br><br>
            <b>Précision :</b> Le modèle atteint une exactitude de <b style="color: #00d4ff;">{:.2f}%</b> sur l'ensemble de test.
        </p>
    </div>
    """.format(accuracy*100), unsafe_allow_html=True)
    
    st.markdown("")
    
    st.markdown("""
    <div style="background-color: #ffffff; border-radius: 12px; padding: 25px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2); margin-top: 20px;">
        <h3 style="color: #1a1a1a; margin-top: 0;">🌺 Les 3 espèces d'Iris</h3>
        <ul style="color: #333333; line-height: 1.8;">
            <li><b style="color: #ff6b6b;">Setosa</b> - Pétales courts et fins</li>
            <li><b style="color: #4ecdc4;">Versicolor</b> - Taille moyenne</li>
            <li><b style="color: #ffd93d;">Virginica</b> - Pétales longs et larges</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# ================================
# PAGE 2 : PRÉDICTION
# ================================

elif page == "🔮 Prédiction":
    st.title("🔮 Prédiction - Classificateur KNN")
    st.markdown("---")
    
    st.markdown("""
    <div style="background-color: #ffffff; border-radius: 12px; padding: 20px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);">
        <h3 style="color: #1a1a1a; margin-top: 0;">📊 Entrez les caractéristiques de la fleur</h3>
        <p style="color: #333333;">Remplissez les champs ci-dessous pour prédire l'espèce de la fleur.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Mesures du sépale**")
        sepal_length = st.number_input(
            "Longueur du sépale (cm)",
            min_value=0.0,
            value=5.1,
            step=0.1
        )
        
        sepal_width = st.number_input(
            "Largeur du sépale (cm)",
            min_value=0.0,
            value=3.5,
            step=0.1
        )
    
    with col2:
        st.markdown("**Mesures du pétale**")
        petal_length = st.number_input(
            "Longueur du pétale (cm)",
            min_value=0.0,
            value=1.4,
            step=0.1
        )
        
        petal_width = st.number_input(
            "Largeur du pétale (cm)",
            min_value=0.0,
            value=0.2,
            step=0.1
        )
    
    st.markdown("")
    
    col_button = st.columns([1, 2, 1])
    with col_button[1]:
        if st.button("🔍 Prédire l'espèce", use_container_width=True):
            input_data = np.array([
                sepal_length,
                sepal_width,
                petal_length,
                petal_width
            ]).reshape(1, -1)
            
            input_scaled = scaler.transform(input_data)
            prediction = knn.predict(input_scaled)[0]
            
            emoji_dict = {
                "setosa": "🌹",
                "versicolor": "🌺",
                "virginica": "🌻"
            }
            
            emoji = emoji_dict.get(prediction, "🌸")
            
            st.success(f"{emoji} Espèce prédite : **{prediction.upper()}**", icon="✅")
            
            st.markdown(f"""
            <div style="background-color: #ffffff; border-radius: 12px; padding: 20px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2); margin-top: 20px;">
                <h3 style="color: #1a1a1a; margin-top: 0;">✨ Résultat de la prédiction</h3>
                <p style="color: #333333; font-size: 18px;"><b>Espèce identifiée :</b> <span style="color: #00d4ff; font-weight: bold;">{prediction.upper()}</span></p>
                <p style="color: #666666; font-size: 14px;">Basé sur les caractéristiques saisies et le modèle KNN entraîné.</p>
            </div>
            """, unsafe_allow_html=True)


# ================================
# PAGE 3 : ANALYSES
# ================================

elif page == "📊 Analyses":
    st.title("📊 Analyses et Visualisations")
    st.markdown("---")
    
    # Exercice 1 : Effectifs des espèces
    st.subheader("Exercice 1 : Effectifs des espèces")
    
    effectifs = df['species'].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Distribution des espèces**")
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('#1a1a1a')
        ax.set_facecolor('#2a2a2a')
        effectifs.plot(kind="bar", color=["#ff6b6b", "#4ecdc4", "#ffd93d"], ax=ax)
        ax.set_title("Histogramme des effectifs", color='white', fontsize=12, fontweight='bold')
        ax.set_xlabel("Espèces", color='white')
        ax.set_ylabel("Effectif", color='white')
        ax.tick_params(colors='white')
        ax.grid(axis="y", alpha=0.2, color='white')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.markdown("**Répartition en secteurs**")
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('#1a1a1a')
        colors_pie = ["#ff6b6b", "#4ecdc4", "#ffd93d"]
        effectifs.plot(kind="pie", autopct="%1.1f%%", ax=ax, colors=colors_pie)
        ax.set_title("Répartition des espèces", color='white', fontsize=12, fontweight='bold')
        ax.set_ylabel("")
        ax.tick_params(colors='white')
        plt.tight_layout()
        st.pyplot(fig)
    
    # Exercice 2 : Variables quantitatives
    st.subheader("Exercice 2 : Analyse des variables quantitatives")
    
    st.markdown("**Résumé statistique de la longueur du pétale**")
    summary_df = df["petal length (cm)"].describe()
    st.write(summary_df)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#1a1a1a')
    ax.set_facecolor('#2a2a2a')
    ax.hist(df["petal length (cm)"], bins=20, edgecolor="white", color="#00d4ff", alpha=0.7)
    ax.set_title("Histogramme de la longueur des pétales", color='white', fontsize=12, fontweight='bold')
    ax.set_xlabel("Longueur du pétale (cm)", color='white')
    ax.set_ylabel("Effectif", color='white')
    ax.tick_params(colors='white')
    ax.grid(axis="y", alpha=0.2, color='white')
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("**Histogrammes des autres variables quantitatives**")
    
    variables = [
        "petal width (cm)",
        "sepal length (cm)",
        "sepal width (cm)"
    ]
    
    cols = st.columns(3)
    colors_hist = ["#ff6b6b", "#4ecdc4", "#ffd93d"]
    
    for idx, var in enumerate(variables):
        with cols[idx]:
            fig, ax = plt.subplots(figsize=(7, 4))
            fig.patch.set_facecolor('#1a1a1a')
            ax.set_facecolor('#2a2a2a')
            ax.hist(df[var], bins=20, edgecolor="white", color=colors_hist[idx], alpha=0.7)
            ax.set_title(f"Histogramme\n{var}", color='white', fontsize=10, fontweight='bold')
            ax.set_xlabel(var, color='white', fontsize=9)
            ax.set_ylabel("Effectif", color='white', fontsize=9)
            ax.tick_params(colors='white', labelsize=8)
            ax.grid(axis="y", alpha=0.2, color='white')
            plt.tight_layout()
            st.pyplot(fig)
    
    # Exercice 3 : Étude bivariée
    st.subheader("Exercice 3 : Étude bivariée")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Pétale : Longueur vs Largeur**")
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('#1a1a1a')
        ax.set_facecolor('#2a2a2a')
        ax.scatter(df["petal length (cm)"], df["petal width (cm)"], alpha=0.6, s=100, color='#00d4ff')
        ax.set_title("Relation : longueur vs largeur du pétale", color='white', fontsize=12, fontweight='bold')
        ax.set_xlabel("Longueur du pétale (cm)", color='white')
        ax.set_ylabel("Largeur du pétale (cm)", color='white')
        ax.tick_params(colors='white')
        ax.grid(alpha=0.2, color='white')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.markdown("**Sépale : Longueur vs Largeur**")
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('#1a1a1a')
        ax.set_facecolor('#2a2a2a')
        ax.scatter(df["sepal length (cm)"], df["sepal width (cm)"], alpha=0.6, s=100, color='#ff6b6b')
        ax.set_title("Relation : longueur vs largeur du sépale", color='white', fontsize=12, fontweight='bold')
        ax.set_xlabel("Longueur du sépale (cm)", color='white')
        ax.set_ylabel("Largeur du sépale (cm)", color='white')
        ax.tick_params(colors='white')
        ax.grid(alpha=0.2, color='white')
        plt.tight_layout()
        st.pyplot(fig)
    
    # Exercice 4 : Boxplots par espèce
    st.subheader("Exercice 4 : Analyse par espèce (Boxplots)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Longueur du pétale selon l'espèce**")
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('#1a1a1a')
        ax.set_facecolor('#2a2a2a')
        df.boxplot(column="petal length (cm)", by="species", ax=ax)
        ax.set_title("Longueur du pétale selon l'espèce", color='white', fontsize=12, fontweight='bold')
        ax.set_xlabel("Espèce", color='white')
        ax.set_ylabel("Longueur du pétale (cm)", color='white')
        ax.tick_params(colors='white')
        plt.suptitle("")
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.markdown("**Largeur du pétale selon l'espèce**")
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('#1a1a1a')
        ax.set_facecolor('#2a2a2a')
        df.boxplot(column="petal width (cm)", by="species", ax=ax)
        ax.set_title("Largeur du pétale selon l'espèce", color='white', fontsize=12, fontweight='bold')
        ax.set_xlabel("Espèce", color='white')
        ax.set_ylabel("Largeur du pétale (cm)", color='white')
        ax.tick_params(colors='white')
        plt.suptitle("")
        plt.tight_layout()
        st.pyplot(fig)
    
    # Exercice 5 : Corrélations et nuage de points par espèce
    st.subheader("Exercice 5 : Corrélations et nuage de points par espèce")
    
    st.markdown("**Matrice de corrélation des variables quantitatives**")
    correlation = df.drop("species", axis=1).corr()
    st.write(correlation)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#1a1a1a')
    sns.heatmap(correlation, annot=True, cmap="coolwarm", center=0, ax=ax, square=True)
    ax.set_title("Matrice de corrélation des variables Iris", color='white', fontsize=12, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("**Nuage de points coloré par espèce**")
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#1a1a1a')
    ax.set_facecolor('#2a2a2a')
    
    colors = {"setosa": "#ff6b6b", "versicolor": "#4ecdc4", "virginica": "#ffd93d"}
    
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
    
    ax.set_title("Nuage de points : longueur vs largeur du pétale par espèce", color='white', fontsize=12, fontweight='bold')
    ax.set_xlabel("Longueur du pétale (cm)", color='white')
    ax.set_ylabel("Largeur du pétale (cm)", color='white')
    ax.legend(facecolor='#2a2a2a', edgecolor='white', labelcolor='white')
    ax.tick_params(colors='white')
    ax.grid(alpha=0.2, color='white')
    plt.tight_layout()
    st.pyplot(fig)


# ================================
# PAGE 4 : DÉTAILS MODÈLE
# ================================

elif page == "📈 Détails Modèle":
    st.title("📈 Détails du Modèle")
    st.markdown("---")
    
    st.markdown("""
    <div style="background-color: #ffffff; border-radius: 12px; padding: 25px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);">
        <h3 style="color: #1a1a1a; margin-top: 0;">🤖 Configuration du modèle K-Nearest Neighbors</h3>
        <ul style="color: #333333; line-height: 1.8; font-size: 16px;">
            <li><b>Algorithme :</b> K-Nearest Neighbors (KNN)</li>
            <li><b>Nombre de voisins (K) :</b> 3</li>
            <li><b>Métrique de distance :</b> Euclidienne (défaut)</li>
            <li><b>Jeu de test :</b> 20% des données</li>
            <li><b>Normalisation :</b> StandardScaler</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Exactitude", f"{accuracy*100:.2f}%")
    
    with col2:
        st.metric("Données d'entraînement", f"{len(X_train)}")
    
    with col3:
        st.metric("Données de test", f"{len(X_test)}")
    
    st.markdown("")
    
    st.markdown("""
    <div style="background-color: #ffffff; border-radius: 12px; padding: 25px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2); margin-top: 20px;">
        <h3 style="color: #1a1a1a; margin-top: 0;">📚 À propos de l'algorithme KNN</h3>
        <p style="color: #333333; line-height: 1.8;">
            <b>K-Nearest Neighbors (KNN)</b> est un algorithme de classification basé sur la proximité. 
            Pour chaque nouvelle fleur à classifier, l'algorithme trouve les K fleurs les plus proches dans l'ensemble 
            d'entraînement et attribue la classe la plus fréquente parmi ces voisins.
        </p>
        <p style="color: #333333; line-height: 1.8;">
            <b>Avantages :</b> Simple, efficace, pas d'entraînement complexe
        </p>
        <p style="color: #333333; line-height: 1.8;">
            <b>Inconvénients :</b> Coûteux en calcul, sensible aux variables avec des échelles différentes
        </p>
    </div>
    """, unsafe_allow_html=True)


# ================================
# PIED DE PAGE
# ================================

st.markdown("---")
st.caption("🚀 Application développée dans le cadre du TP de classification des fleurs Iris – Apprentissage automatique.")
