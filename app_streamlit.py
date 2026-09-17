import os

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Prédiction violence physique conjugale",
    page_icon=":material/shield:",
    layout="wide",
)


def charger_css(chemin: str) -> None:
    if os.path.exists(chemin):
        with open(chemin, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


charger_css("style.css")

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
    "245": "Boulou", "256": "Maka / Makya", "996": "Autre ethnie",
    "Autre": "Autre ethnie",
}

OCCUPATION_LABELS = {
    "11": "Agriculteur / cultivateur", "33": "Enseignant",
    "62": "Commerçant / vendeur", "66": "Conducteur de véhicule",
    "72": "Métiers du bâtiment", "73": "Métallurgie / mécanique",
    "82": "Gendarmerie / armée", "96": "Autre métier",
    "Autre": "Autre métier",
    "Manquant": "Sans partenaire actuel / non renseigné",
}

RELIGION_LABELS = {
    "1": "Catholique", "2": "Protestant", "3": "Autres chrétiens", "4": "Musulman",
    "Autre": "Autre / Animiste / Sans religion",
}

RESIDENCE_LABELS = {
    "1": "Oui, il vit avec elle", "2": "Non, il vit ailleurs", "Manquant": "Non renseigné",
}

EDUCATION_LABELS = {0: "Aucune", 1: "École primaire", 2: "École secondaire", 3: "Études supérieures"}

LABELS_PAR_VARIABLE = {
    "region": REGION_LABELS,
    "ethnicite": ETHNICITE_LABELS,
    "occupation_partenaire": OCCUPATION_LABELS,
    "religion": RELIGION_LABELS,
    "residence_partenaire": RESIDENCE_LABELS,
}


def libelle(colonne, code):
    return LABELS_PAR_VARIABLE.get(colonne, {}).get(code, f"Autre ({code})")


def _index_slider(question: str, options: list) -> int:
    """Curseur à choix textuels : renvoie l'index du choix (utilisé par le modèle)."""
    choix = st.select_slider(question, options=options, value=options[0])
    return options.index(choix)


# ---------------------------------------------------------------------------
# ICONES (SVG inline, sans emoji) POUR LE HTML PERSONNALISE
# ---------------------------------------------------------------------------
_ICON_PATHS = {
    "shield": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>',
    "user": '<path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
    "users": (
        '<path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/>'
        '<path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>'
    ),
    "home": '<path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/>',
    "flag": '<path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z"/><line x1="4" y1="22" x2="4" y2="15"/>',
    "warning": (
        '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>'
        '<line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>'
    ),
    "check": '<polyline points="9 11 12 14 22 4"/><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/>',
}


def icon(name: str) -> str:
    return (
        '<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{_ICON_PATHS[name]}</svg>'
    )


# ---------------------------------------------------------------------------
# 1. CHARGEMENT DU MODELE PRE-ENTRAINE (mis en cache)
# ---------------------------------------------------------------------------
MODEL_PATH = "model.joblib"

CAT_VARS = ["ethnicite", "religion", "region", "occupation_partenaire", "residence_partenaire"]
CIBLE = "violence_physique"


@st.cache_resource
def charger_modele():
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"Fichier modèle introuvable ({MODEL_PATH}). "
            "Lancez d'abord `python train_model.py` pour entraîner et sauvegarder le modèle."
        )
        st.stop()
    bundle = joblib.load(MODEL_PATH)
    return (
        bundle["model"],
        bundle["colonnes_entrainement"],
        bundle["medianes"],
        bundle["categories_valides"],
        bundle["num_vars"],
    )


model, colonnes_entrainement, medianes, categories_valides, num_vars = charger_modele()

# ---------------------------------------------------------------------------
# 2. EN-TETE
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero">
        <span class="brand-pill">{icon("shield")} SocaStat</span>
        <h1>Est-ce que ma situation présente un risque ?</h1>
        <p>
            Répondez à quelques questions simples pour obtenir une estimation, à titre
            indicatif, du niveau de risque de violence physique dans un couple.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# 3. ONGLETS : A PROPOS / PREDICTION
# ---------------------------------------------------------------------------
tab_apropos, tab_prediction = st.tabs([":material/info: À propos", ":material/search: Faire une estimation"])

with tab_apropos:
    st.markdown(
        f"""
        <div class="about-card">
            <h3>{icon("flag")} Le but de cet outil</h3>
            <p>
                Cette application aide à réfléchir sur une situation de couple.
                En répondant à quelques questions simples, vous obtenez une estimation
                du niveau de risque de violence physique — <b>faible</b>, <b>modéré</b>
                ou <b>élevé</b>.
            </p>
            <hr class="about-divider">
            <h3>{icon("users")} Réalisé par</h3>
            <div>
                <span class="author-chip">Telesphore Ebanga-Mballa</span>
                <span class="author-chip">Stéphane Bella-Mbarga</span>
                <span class="author-chip">Brenda Enow</span>
                <span class="author-chip">Kum-Collins</span>
                <span class="author-chip">Georges Nguefack-Tsague</span>
            </div>
            <hr class="about-divider">
            <h3>{icon("warning")} Ce qu'il faut savoir</h3>
            <p>
                Le résultat affiché est une estimation basée sur des situations déjà
                observées par le passé. <b>Ce n'est pas un diagnostic</b> et cela ne
                remplace pas l'avis d'un médecin, d'un travailleur social ou de toute
                autre personne qualifiée. En cas de danger, rapprochez-vous des
                autorités locales ou d'une association d'aide aux victimes.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with tab_prediction:
    # -----------------------------------------------------------------------
    # 3.1 FORMULAIRE DE SAISIE
    # -----------------------------------------------------------------------
    with st.form("formulaire_prediction"):
        col1, col2, col3 = st.columns(3)

        with col1:
            with st.container(border=True):
                st.markdown(f'<div class="section-title">{icon("user")} La femme</div>', unsafe_allow_html=True)
                age_femme = st.slider("Son âge", 15, 49, 28)
                age_premiere_union = st.slider("Son âge lors de sa première mise en couple", 10, 40, 18)
                age_premier_rapport = st.slider("Son âge lors de son premier rapport intime", 10, 40, 17)
                education_femme = st.selectbox(
                    "Son niveau d'études", [0, 1, 2, 3], index=1,
                    format_func=lambda x: EDUCATION_LABELS[x],
                    key="education_femme",
                )
                richesse = st.select_slider(
                    "Le niveau de vie de son foyer",
                    options=[1, 2, 3, 4, 5], value=3,
                    format_func=lambda x: {
                        1: "Très modeste", 2: "Modeste", 3: "Moyen", 4: "Aisé", 5: "Riche",
                    }[x],
                )
                milieu = st.radio(
                    "Où vit-elle ?", [0, 1],
                    format_func=lambda x: "En ville" if x == 0 else "À la campagne", horizontal=True
                )

        with col2:
            with st.container(border=True):
                st.markdown(f'<div class="section-title">{icon("user")} Le partenaire</div>', unsafe_allow_html=True)
                age_partenaire = st.slider("Son âge", 15, 90, 35)
                education_partenaire = st.selectbox(
                    "Son niveau d'études", [0, 1, 2, 3], index=1,
                    format_func=lambda x: EDUCATION_LABELS[x],
                    key="education_partenaire",
                )
                alcool_partenaire = st.selectbox(
                    "Boit-il de l'alcool ?", [0, 1, 2],
                    format_func=lambda x: {0: "Jamais", 1: "Souvent", 2: "Parfois"}[x]
                )
                ctrl_score = _index_slider(
                    "Contrôle-t-il ses activités (sorties, argent, amis) ?",
                    ["Pas du tout", "Un peu", "Modérément", "Assez", "Beaucoup", "Énormément"],
                )
                polygamie = _index_slider(
                    "A-t-il d'autres épouses ?",
                    ["Aucune", "1 autre", "2 autres", "3 autres"],
                )

        with col3:
            with st.container(border=True):
                st.markdown(f'<div class="section-title">{icon("home")} Le couple et la famille</div>', unsafe_allow_html=True)
                transmission_intergen = st.radio(
                    "Le père de la femme battait-il sa mère quand elle était enfant ?", [0, 1],
                    format_func=lambda x: "Non" if x == 0 else "Oui", horizontal=True
                )
                attitude_score = _index_slider(
                    "Pense-t-elle que frapper sa femme peut parfois être justifié ?",
                    ["Jamais", "Rarement", "Parfois", "Souvent", "La plupart du temps", "Toujours"],
                )
                autonomie_score = _index_slider(
                    "Décide-t-elle elle-même pour l'argent, sa santé et ses sorties ?",
                    ["Rarement", "Parfois", "Souvent", "Toujours"],
                )
                region = st.selectbox(
                    "Région où vit le couple", sorted(categories_valides["region"], key=lambda c: int(c)),
                    format_func=lambda c: libelle("region", c)
                )
                ethnicite = st.selectbox(
                    "Ethnie de la femme", categories_valides["ethnicite"],
                    format_func=lambda c: libelle("ethnicite", c)
                )
                religion = st.selectbox(
                    "Religion de la femme", categories_valides["religion"],
                    format_func=lambda c: libelle("religion", c)
                )
                occupation_partenaire = st.selectbox(
                    "Métier du partenaire", categories_valides["occupation_partenaire"],
                    format_func=lambda c: libelle("occupation_partenaire", c)
                )
                residence_partenaire = st.selectbox(
                    "Le partenaire vit-il avec elle actuellement ?", categories_valides["residence_partenaire"],
                    format_func=lambda c: libelle("residence_partenaire", c)
                )

        st.markdown("<br>", unsafe_allow_html=True)
        valider = st.form_submit_button(
            "Voir le résultat", icon=":material/search:", type="primary", use_container_width=True
        )

    # -----------------------------------------------------------------------
    # 3.2 CONSTRUCTION DE L'OBSERVATION ET PREDICTION
    # -----------------------------------------------------------------------
    if valider:
        ecart_age = age_partenaire - age_femme

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
        proba_pct = min(proba, 1.0) * 100

        if proba < 0.25:
            niveau, classe = "Faible", "low"
        elif proba < 0.50:
            niveau, classe = "Modéré", "medium"
        else:
            niveau, classe = "Élevé", "high"

        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="section-title">{icon("check")} Résultat</div>', unsafe_allow_html=True)

        c1, c2 = st.columns([1, 2])

        with c1:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="value">{proba:.0%}</div>
                    <div class="label">Chances estimées</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c2:
            st.markdown(
                f"""
                <p>Niveau de risque : <span class="badge badge-{classe}">{niveau}</span></p>
                <div class="gauge-track">
                    <div class="gauge-fill {classe}" style="width: {proba_pct:.1f}%;"></div>
                </div>
                <div class="gauge-caption"><span>Faible</span><span>Modéré</span><span>Élevé</span></div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

        st.caption(
            "Ce résultat est une estimation à titre indicatif. Il ne remplace pas "
            "l'avis d'un professionnel (médecin, travailleur social, association d'aide)."
        )

# ---------------------------------------------------------------------------
# 4. PIED DE PAGE
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-footer">
        Cet outil aide à la réflexion — il ne remplace pas un avis médical ou social professionnel.
    </div>
    """,
    unsafe_allow_html=True,
)
