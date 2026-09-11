import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Prediction violence physique conjugale", layout="wide")

# ---------------------------------------------------------------------------
# LIBELLES REELS (dictionnaire de variables EDS Cameroun) POUR L'AFFICHAGE
# Les codes internes restent inchanges (utilises pour l'entrainement du modele),
# seul l'affichage a l'ecran montre le nom plutot que le code.
# ---------------------------------------------------------------------------
REGION_LABELS = {
    "1": "Adamaoua", "2": "Centre (hors Yaoundé)", "3": "Douala", "4": "Est",
    "5": "Extrême-Nord", "6": "Littoral (hors Douala)", "7": "Nord",
    "8": "Nord-Ouest", "9": "Ouest", "10": "Sud", "11": "Sud-Ouest", "12": "Yaoundé",
}

ETHNICITE_LABELS = {
    "3": "Foulbé", "78": "Gbaya", "101": "Toupouri", "176": "Bamoun",
    "202": "Bamiléké", "232": "Bassa", "235": "Éton", "236": "Ewondo",
    "245": "Boulou", "256": "Maka / Makya", "996": "Autre (code EDS)",
    "Autre": "Autres ethnies (regroupées, effectif < 2%)",
}

OCCUPATION_LABELS = {
    "11": "Agriculteur / cultivateur", "33": "Enseignant",
    "62": "Commerçant / vendeur", "66": "Conducteur de véhicule",
    "72": "Métiers du bâtiment", "73": "Métallurgie / construction mécanique",
    "82": "Gendarmerie / armée", "96": "Autre profession (code EDS)",
    "Autre": "Autres professions (regroupées, effectif < 2%)",
    "Manquant": "Sans partenaire actuel / non renseigné",
}

RELIGION_LABELS = {
    "1": "Catholique", "2": "Protestant", "3": "Autres chrétiens", "4": "Musulman",
    "Autre": "Autre / Animiste / Sans religion",
}

RESIDENCE_LABELS = {
    "1": "Vit avec elle", "2": "Vit ailleurs", "Manquant": "Non renseigné",
}

EDUCATION_LABELS = {0: "Aucune", 1: "Primaire", 2: "Secondaire", 3: "Supérieur"}

LABELS_PAR_VARIABLE = {
    "region": REGION_LABELS,
    "ethnicite": ETHNICITE_LABELS,
    "occupation_partenaire": OCCUPATION_LABELS,
    "religion": RELIGION_LABELS,
    "residence_partenaire": RESIDENCE_LABELS,
}


def libelle(colonne, code):
    return LABELS_PAR_VARIABLE.get(colonne, {}).get(code, f"Code EDS {code}")

# ---------------------------------------------------------------------------
# 1. CHARGEMENT ET ENTRAINEMENT DU MODELE (mis en cache)
# ---------------------------------------------------------------------------
DATA_PATH = "dataset_violence.csv"   # <-- adapte le chemin si besoin

CAT_VARS = ["ethnicite", "religion", "region", "occupation_partenaire", "residence_partenaire"]
CIBLE = "violence_physique"
SEUIL_RARE = 0.02


@st.cache_resource
def entrainer_modele():
    df = pd.read_csv(DATA_PATH)
    num_vars = [c for c in df.columns if c not in CAT_VARS + [CIBLE]]

    medianes = {}
    for col in num_vars:
        if df[col].isna().sum() > 0:
            df[f"{col}_manquant"] = df[col].isna().astype(int)
            medianes[col] = df[col].median()
            df[col] = df[col].fillna(medianes[col])
        else:
            medianes[col] = df[col].median()

    categories_valides = {}
    for col in CAT_VARS:
        df[col] = df[col].fillna(-1).astype(int).astype(str).replace("-1", "Manquant")
        freq = df[col].value_counts(normalize=True)
        rares = freq[freq < SEUIL_RARE].index
        df[col] = df[col].apply(lambda x: "Autre" if x in rares else x)
        categories_valides[col] = sorted(df[col].unique().tolist())

    df_encoded = pd.get_dummies(df, columns=CAT_VARS, prefix=CAT_VARS)
    bool_cols = df_encoded.select_dtypes(include="bool").columns
    df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

    X = df_encoded.drop(columns=[CIBLE])
    y = df_encoded[CIBLE]

    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    model.fit(X, y)

    return model, X.columns.tolist(), medianes, categories_valides, num_vars


model, colonnes_entrainement, medianes, categories_valides, num_vars = entrainer_modele()

# ---------------------------------------------------------------------------
# 2. INTERFACE - SAISIE DES CARACTERISTIQUES
# ---------------------------------------------------------------------------
st.title("Prédiction du risque de violence physique conjugale")
st.markdown("""
**Auteurs :**  
- EBANGA MBALLA
- BELLA MBARGA
- ENOW
- Kum Collins
- Georges Nguefack-Tsague
""")
st.caption(
    "Outil d'aide à la décision basé sur un modèle de régression logistique "
    "entraîné sur les données EDS Cameroun (module violence domestique). "
    "Ne constitue pas un diagnostic — à utiliser en complément du jugement professionnel."
)

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Femme")
    age_femme = st.slider("Âge de la répondante", 15, 49, 28)
    age_premiere_union = st.slider("Âge à la première union", 10, 40, 18)
    age_premier_rapport = st.slider("Âge au premier rapport sexuel", 10, 40, 17)
    education_femme = st.selectbox(
        "Niveau d'éducation de la femme", [0, 1, 2, 3], index=1,
        format_func=lambda x: EDUCATION_LABELS[x]
    )
    richesse = st.selectbox("Quintile de richesse", [1, 2, 3, 4, 5], index=2,
                             format_func=lambda x: {1: "1 - Le plus pauvre", 5: "5 - Le plus riche"}.get(x, str(x)))
    milieu = st.radio("Milieu de résidence", [0, 1], format_func=lambda x: "Urbain" if x == 0 else "Rural", horizontal=True)

with col2:
    st.subheader("Partenaire")
    age_partenaire = st.slider("Âge du partenaire", 15, 90, 35)
    education_partenaire = st.selectbox(
        "Niveau d'éducation du partenaire", [0, 1, 2, 3], index=1,
        format_func=lambda x: EDUCATION_LABELS[x]
    )
    alcool_partenaire = st.selectbox("Consommation d'alcool du partenaire", [0, 1, 2],
                                      format_func=lambda x: {0: "Jamais", 1: "Souvent", 2: "Parfois"}[x])
    ctrl_score = st.slider("Score de comportements de contrôle (0 = aucun, 5 = maximal)", 0, 5, 0)
    polygamie = st.selectbox("Situation de polygamie (nombre de co-épouses)", [0, 1, 2, 3], index=0)

with col3:
    st.subheader("Contexte / couple")
    transmission_intergen = st.radio("Le père de la répondante battait-il sa mère ?", [0, 1],
                                      format_func=lambda x: "Non" if x == 0 else "Oui", horizontal=True)
    attitude_score = st.slider("Score d'attitudes justifiant la violence (0 à 5)", 0, 5, 0)
    autonomie_score = st.slider("Score d'autonomie décisionnelle (0 à 3)", 0, 3, 1)
    region = st.selectbox(
        "Région", sorted(categories_valides["region"], key=lambda c: int(c)),
        format_func=lambda c: libelle("region", c)
    )
    ethnicite = st.selectbox(
        "Ethnicité", categories_valides["ethnicite"],
        format_func=lambda c: libelle("ethnicite", c)
    )
    religion = st.selectbox(
        "Religion", categories_valides["religion"],
        format_func=lambda c: libelle("religion", c)
    )
    occupation_partenaire = st.selectbox(
        "Occupation du partenaire", categories_valides["occupation_partenaire"],
        format_func=lambda c: libelle("occupation_partenaire", c)
    )
    residence_partenaire = st.selectbox(
        "Réside actuellement avec le partenaire", categories_valides["residence_partenaire"],
        format_func=lambda c: libelle("residence_partenaire", c)
    )

ecart_age = age_partenaire - age_femme

st.divider()

# ---------------------------------------------------------------------------
# 3. CONSTRUCTION DE L'OBSERVATION ET PREDICTION
# ---------------------------------------------------------------------------
if st.button("Calculer le risque", type="primary", use_container_width=True):

    observation = {
        "ctrl_score": ctrl_score,
        "transmission_intergen": transmission_intergen,
        "alcool_partenaire": alcool_partenaire,
        "ecart_age": ecart_age,
        "age_femme": age_femme,
        "age_partenaire": age_partenaire,
        "age_premiere_union": age_premiere_union,
        "age_premier_rapport": age_premier_rapport,
        "richesse": richesse,
        "education_partenaire": education_partenaire,
        "education_femme": education_femme,
        "attitude_score": attitude_score,
        "autonomie_score": autonomie_score,
        "polygamie": polygamie,
        "milieu": milieu,
        "ethnicite": ethnicite,
        "religion": religion,
        "region": region,
        "occupation_partenaire": occupation_partenaire,
        "residence_partenaire": residence_partenaire,
    }

    df_obs = pd.DataFrame([observation])

    # indicateurs de valeur manquante (aucune ici, saisie complete -> tous a 0)
    for col in num_vars:
        col_manquant = f"{col}_manquant"
        if col_manquant in colonnes_entrainement:
            df_obs[col_manquant] = 0

    # encodage one-hot des variables categorielles, aligne sur les colonnes d'entrainement
    df_obs_encoded = pd.get_dummies(df_obs, columns=CAT_VARS, prefix=CAT_VARS)
    df_obs_encoded = df_obs_encoded.reindex(columns=colonnes_entrainement, fill_value=0)

    proba = model.predict_proba(df_obs_encoded)[0, 1]

    st.subheader("Résultat")
    c1, c2 = st.columns([1, 2])

    with c1:
        st.metric("Probabilité prédite", f"{proba:.1%}")

    with c2:
        if proba < 0.25:
            st.success("Risque prédit : **Faible**")
        elif proba < 0.50:
            st.warning("Risque prédit : **Modéré**")
        else:
            st.error("Risque prédit : **Élevé**")

    st.progress(min(proba, 1.0))

    st.caption(
        "Cette estimation reflète des associations statistiques observées dans l'échantillon "
        "EDS et ne doit pas être interprétée comme un diagnostic individuel ni une preuve de causalité."
    )
