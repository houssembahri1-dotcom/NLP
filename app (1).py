"""
=============================================================
🚀 Déploiement Streamlit - Classification Spam / Ham
Cours Machine Learning - NLP Use Case
=============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import zipfile
import os
import matplotlib.pyplot as plt
from collections import Counter

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# -----------------------------------------------------------
# Configuration de la page
# -----------------------------------------------------------
st.set_page_config(
    page_title="Spam Classifier - NLP",
    page_icon="📩",
    layout="wide"
)

# -----------------------------------------------------------
# Téléchargement NLTK
# -----------------------------------------------------------
@st.cache_resource
def download_nltk():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

download_nltk()

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# -----------------------------------------------------------
# Fonction de preprocessing
# -----------------------------------------------------------
def preprocessing(text):
    """Pipeline de prétraitement du texte"""
    text = re.sub("[^a-zA-Z]", " ", str(text))
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [word for word in tokens if len(word) >= 3]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# -----------------------------------------------------------
# Chargement et entraînement du modèle (caché en cache)
# -----------------------------------------------------------
@st.cache_resource
def load_and_train():
    """Charge les données, entraîne les modèles et retourne tout"""

    # Télécharger le dataset si nécessaire
    if not os.path.exists("SMSSpamCollection"):
        import urllib.request
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
        urllib.request.urlretrieve(url, "smsspamcollection.zip")
        with zipfile.ZipFile("smsspamcollection.zip", "r") as z:
            z.extractall(".")

    # Charger
    df = pd.read_csv("SMSSpamCollection", sep="\t", header=None, names=["Y", "X"])

    # Preprocessing
    df["X_processed"] = df["X"].apply(preprocessing)
    df = df.dropna(subset=["X_processed"])
    df = df[df["X_processed"].str.strip() != ""]

    # Split
    X = df["X_processed"]
    Y = df["Y"]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Bag of Words
    bow_vec = CountVectorizer()
    X_train_bow = bow_vec.fit_transform(X_train)
    X_test_bow = bow_vec.transform(X_test)
    nb_bow = MultinomialNB()
    nb_bow.fit(X_train_bow, Y_train)
    Y_pred_bow = nb_bow.predict(X_test_bow)

    # TF-IDF
    tfidf_vec = TfidfVectorizer()
    X_train_tfidf = tfidf_vec.fit_transform(X_train)
    X_test_tfidf = tfidf_vec.transform(X_test)
    nb_tfidf = MultinomialNB()
    nb_tfidf.fit(X_train_tfidf, Y_train)
    Y_pred_tfidf = nb_tfidf.predict(X_test_tfidf)

    return {
        "df": df,
        "bow_vec": bow_vec, "nb_bow": nb_bow,
        "tfidf_vec": tfidf_vec, "nb_tfidf": nb_tfidf,
        "X_test": X_test, "Y_test": Y_test,
        "Y_pred_bow": Y_pred_bow, "Y_pred_tfidf": Y_pred_tfidf,
        "X_train": X_train, "Y_train": Y_train
    }

data = load_and_train()

# -----------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/04/ChatGPT_logo.svg/120px-ChatGPT_logo.svg.png", width=60)
st.sidebar.title("📩 Spam Classifier")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Accueil", "📊 Exploration (EDA)", "⚙️ Preprocessing", "🤖 Modèle & Résultats", "🔍 Prédiction"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Cours Machine Learning**")
st.sidebar.markdown("Aurélien Vannieuwenhuyze")
st.sidebar.markdown("Université catholique de Lille")

# -----------------------------------------------------------
# PAGE : ACCUEIL
# -----------------------------------------------------------
if page == "🏠 Accueil":
    st.title("📩 Classification Spam / Ham avec NLP")
    st.markdown("### Use Case — Cours Machine Learning")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total SMS", f"{len(data['df']):,}")
    with col2:
        ham_count = len(data['df'][data['df']['Y'] == 'ham'])
        st.metric("Ham (légitimes)", f"{ham_count:,}")
    with col3:
        spam_count = len(data['df'][data['df']['Y'] == 'spam'])
        st.metric("Spam", f"{spam_count:,}")

    st.markdown("---")

    st.markdown("""
    #### 📋 Pipeline du projet

    | Étape | Description |
    |-------|-------------|
    | **1** | Récupération des données (SMS Spam Collection - UCI) |
    | **2** | Exploration Data Analysis |
    | **3** | Text Preprocessing (regex, tokenization, stopwords, lemmatisation) |
    | **4** | Application du preprocessing sur tout le dataset |
    | **5** | Analyse de fréquence des mots par classe |
    | **6** | Machine Learning : Bag of Words & TF-IDF + Naive Bayes |

    #### 🧠 Origine du mot SPAM
    Le terme **SPAM** vient de la marque de viande en conserve "**Sp**iced h**am**" (faux jambon).
    L'usage pour les messages indésirables est inspiré d'un sketch des **Monty Python** où le mot "spam" est répété sans cesse.
    """)

# -----------------------------------------------------------
# PAGE : EDA
# -----------------------------------------------------------
elif page == "📊 Exploration (EDA)":
    st.title("📊 Exploration Data Analysis")

    df = data["df"]

    st.markdown("### Aperçu du dataset")
    st.dataframe(df[["X", "Y"]].head(10), use_container_width=True)

    st.markdown("### Répartition des classes")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(5, 5))
        counts = df["Y"].value_counts()
        colors = ["#2ecc71", "#e74c3c"]
        ax.pie(counts.values, labels=counts.index.str.upper(), autopct='%1.1f%%',
               colors=colors, startangle=90, textprops={'fontsize': 14})
        ax.set_title("Distribution Ham / Spam", fontsize=14, fontweight='bold')
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.bar(counts.index.str.upper(), counts.values, color=colors, edgecolor='black')
        ax.set_ylabel("Nombre de SMS")
        ax.set_title("Nombre de SMS par classe", fontsize=14, fontweight='bold')
        for i, v in enumerate(counts.values):
            ax.text(i, v + 50, str(v), ha='center', fontweight='bold')
        st.pyplot(fig)

    st.markdown("### Fréquence des mots par classe")

    def get_word_freq(label, top_n=30):
        texts = df[df["Y"] == label]["X_processed"]
        all_words = " ".join(texts).split()
        return Counter(all_words).most_common(top_n)

    tab1, tab2 = st.tabs(["🔴 SPAM", "🟢 HAM"])

    with tab1:
        freq = get_word_freq("spam")
        words = [w for w, c in freq]
        counts_w = [c for w, c in freq]
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(words, counts_w, color='#e74c3c', marker='o', markersize=4)
        ax.set_title("SPAM — Fréquence des mots", fontsize=14, fontweight='bold')
        ax.set_xlabel("Mots")
        ax.set_ylabel("Fréquence")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

    with tab2:
        freq = get_word_freq("ham")
        words = [w for w, c in freq]
        counts_w = [c for w, c in freq]
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(words, counts_w, color='#2ecc71', marker='o', markersize=4)
        ax.set_title("HAM — Fréquence des mots", fontsize=14, fontweight='bold')
        ax.set_xlabel("Mots")
        ax.set_ylabel("Fréquence")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

# -----------------------------------------------------------
# PAGE : PREPROCESSING
# -----------------------------------------------------------
elif page == "⚙️ Preprocessing":
    st.title("⚙️ Text Preprocessing")

    st.markdown("""
    ### Pipeline de prétraitement

    ```python
    def preprocessing(text):
        text = re.sub("[^a-zA-Z]", " ", str(text))   # 1. Supprimer chiffres/caractères spéciaux
        tokens = word_tokenize(text)                   # 2. Tokenization
        tokens = [word.lower() for word in tokens]     # 3. Mise en minuscule
        tokens = [w for w in tokens if w not in stop_words]  # 4. Supprimer stop words
        tokens = [w for w in tokens if len(w) >= 3]    # 5. Supprimer mots < 3 caractères
        tokens = [lemmatizer.lemmatize(w) for w in tokens]   # 6. Lemmatisation
        return " ".join(tokens)                        # 7. Reconstruire la phrase
    ```
    """)

    st.markdown("### 🧪 Testez le preprocessing")

    default_text = "Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat…"
    user_input = st.text_area("Entrez un texte à prétraiter :", value=default_text, height=100)

    if st.button("▶️ Lancer le preprocessing", type="primary"):
        result = preprocessing(user_input)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Texte original :**")
            st.info(user_input)
        with col2:
            st.markdown("**Après preprocessing :**")
            st.success(result)

        st.markdown("### Détail étape par étape")

        steps = []
        # Step 1
        t1 = re.sub("[^a-zA-Z]", " ", str(user_input))
        steps.append(("1. Regex `[^a-zA-Z]` → espace", t1))
        # Step 2
        tokens = word_tokenize(t1)
        steps.append(("2. Tokenization", str(tokens)))
        # Step 3
        tokens = [w.lower() for w in tokens]
        steps.append(("3. Mise en minuscule", str(tokens)))
        # Step 4
        tokens = [w for w in tokens if w not in stop_words]
        steps.append(("4. Suppression stop words", str(tokens)))
        # Step 5
        tokens = [w for w in tokens if len(w) >= 3]
        steps.append(("5. Mots ≥ 3 caractères", str(tokens)))
        # Step 6
        tokens = [lemmatizer.lemmatize(w) for w in tokens]
        steps.append(("6. Lemmatisation", str(tokens)))
        # Step 7
        steps.append(("7. Reconstruction", " ".join(tokens)))

        for step_name, step_result in steps:
            st.markdown(f"**{step_name}**")
            st.code(step_result)

# -----------------------------------------------------------
# PAGE : MODÈLE & RÉSULTATS
# -----------------------------------------------------------
elif page == "🤖 Modèle & Résultats":
    st.title("🤖 Modèle & Résultats")

    Y_test = data["Y_test"]
    Y_pred_bow = data["Y_pred_bow"]
    Y_pred_tfidf = data["Y_pred_tfidf"]

    # Métriques globales
    acc_bow = accuracy_score(Y_test, Y_pred_bow)
    acc_tfidf = accuracy_score(Y_test, Y_pred_tfidf)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("🎯 Accuracy — Bag of Words", f"{acc_bow:.2%}")
    with col2:
        st.metric("🎯 Accuracy — TF-IDF", f"{acc_tfidf:.2%}")

    st.markdown("---")

    method = st.selectbox("Choisissez la méthode de vectorisation :", ["Bag of Words", "TF-IDF"])

    if method == "Bag of Words":
        Y_pred = Y_pred_bow
    else:
        Y_pred = Y_pred_tfidf

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Matrice de confusion")
        cm = confusion_matrix(Y_test, Y_pred, labels=["ham", "spam"])
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["HAM", "SPAM"], fontsize=12)
        ax.set_yticklabels(["HAM", "SPAM"], fontsize=12)
        ax.set_xlabel("Prédit", fontsize=13)
        ax.set_ylabel("Réel", fontsize=13)
        ax.set_title(f"Matrice de confusion — {method}", fontsize=14, fontweight='bold')
        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=18, color=color, fontweight='bold')
        fig.colorbar(im)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.markdown("### Rapport de classification")
        report = classification_report(Y_test, Y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)

    st.markdown("---")
    st.markdown("### Comparaison des deux méthodes")

    comp_df = pd.DataFrame({
        "Méthode": ["Bag of Words + Naive Bayes", "TF-IDF + Naive Bayes"],
        "Accuracy": [acc_bow, acc_tfidf],
        "Precision (spam)": [
            classification_report(Y_test, Y_pred_bow, output_dict=True)["spam"]["precision"],
            classification_report(Y_test, Y_pred_tfidf, output_dict=True)["spam"]["precision"]
        ],
        "Recall (spam)": [
            classification_report(Y_test, Y_pred_bow, output_dict=True)["spam"]["recall"],
            classification_report(Y_test, Y_pred_tfidf, output_dict=True)["spam"]["recall"]
        ],
        "F1-Score (spam)": [
            classification_report(Y_test, Y_pred_bow, output_dict=True)["spam"]["f1-score"],
            classification_report(Y_test, Y_pred_tfidf, output_dict=True)["spam"]["f1-score"]
        ]
    })
    st.dataframe(comp_df.style.format({
        "Accuracy": "{:.4f}",
        "Precision (spam)": "{:.4f}",
        "Recall (spam)": "{:.4f}",
        "F1-Score (spam)": "{:.4f}"
    }), use_container_width=True)

# -----------------------------------------------------------
# PAGE : PRÉDICTION
# -----------------------------------------------------------
elif page == "🔍 Prédiction":
    st.title("🔍 Testez le classifieur")
    st.markdown("Entrez un message SMS et le modèle prédit s'il s'agit d'un **Spam** ou d'un **Ham** (légitime).")

    st.markdown("---")

    method = st.radio("Méthode de vectorisation :", ["Bag of Words", "TF-IDF"], horizontal=True)

    if method == "Bag of Words":
        vectorizer = data["bow_vec"]
        model = data["nb_bow"]
    else:
        vectorizer = data["tfidf_vec"]
        model = data["nb_tfidf"]

    sentence = st.text_input(
        "📝 Votre message SMS :",
        placeholder="Ex: Congratulations! You've won a FREE prize. Call now!"
    )

    # Exemples rapides
    st.markdown("**Exemples rapides :**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("🔴 Spam 1"):
            sentence = "CONGRATULATIONS! You've won a FREE iPhone! Call now!"
    with col2:
        if st.button("🔴 Spam 2"):
            sentence = "FREE entry in 2 a weekly competition to win FA Cup"
    with col3:
        if st.button("🟢 Ham 1"):
            sentence = "Hey, are you coming to the party tonight?"
    with col4:
        if st.button("🟢 Ham 2"):
            sentence = "Can you pick up some milk on your way home?"

    if sentence:
        processed = preprocessing(sentence)
        vectorized = vectorizer.transform([processed])
        prediction = model.predict(vectorized)[0]
        proba = model.predict_proba(vectorized)[0]

        st.markdown("---")

        col1, col2 = st.columns([1, 2])

        with col1:
            if prediction == "spam":
                st.error("## 🚫 SPAM")
            else:
                st.success("## ✅ HAM")

            st.markdown(f"**Confiance :** `{max(proba):.2%}`")

        with col2:
            st.markdown("**Message original :**")
            st.info(sentence)
            st.markdown("**Après preprocessing :**")
            st.code(processed)

            # Barre de probabilité
            st.markdown("**Probabilités :**")
            prob_df = pd.DataFrame({
                "Classe": ["Ham", "Spam"],
                "Probabilité": [proba[0], proba[1]]
            })

            fig, ax = plt.subplots(figsize=(8, 1.5))
            colors = ["#2ecc71", "#e74c3c"]
            bars = ax.barh(prob_df["Classe"], prob_df["Probabilité"], color=colors, edgecolor='black', height=0.5)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probabilité")
            for bar, val in zip(bars, prob_df["Probabilité"]):
                ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                        f'{val:.4f}', va='center', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
