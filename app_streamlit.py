import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

import train_model

st.set_page_config(
    page_title="Intimate Partner Violence Risk Prediction",
    page_icon=":material/shield:",
    layout="wide",
)


def charger_css(chemin: str) -> None:
    if os.path.exists(chemin):
        with open(chemin, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


charger_css("style.css")

# ---------------------------------------------------------------------------
# LANGUAGE TOGGLE (default: English, switchable to French)
# ---------------------------------------------------------------------------
if "lang" not in st.session_state:
    st.session_state.lang = "en"

_top_spacer, _top_switch = st.columns([8, 1])
with _top_switch:
    _target_lang = "fr" if st.session_state.lang == "en" else "en"
    _target_label = "Français" if st.session_state.lang == "en" else "English"
    if st.button(_target_label, use_container_width=True):
        st.session_state.lang = _target_lang
        st.rerun()

lang = st.session_state.lang

# ---------------------------------------------------------------------------
# DISPLAY LABELS (EDS Cameroon variable dictionary), per language
# The internal codes stay unchanged (used to train the model), only the
# on-screen display shows the name instead of the raw code.
# ---------------------------------------------------------------------------
REGION_LABELS = {
    "en": {
        "1": "Adamawa", "2": "Centre (excluding Yaoundé)", "3": "Douala", "4": "East",
        "5": "Far North", "6": "Littoral (excluding Douala)", "7": "North",
        "8": "North-West", "9": "West", "10": "South", "11": "South-West", "12": "Yaoundé",
    },
    "fr": {
        "1": "Adamaoua", "2": "Centre (hors Yaoundé)", "3": "Douala", "4": "Est",
        "5": "Extrême-Nord", "6": "Littoral (hors Douala)", "7": "Nord",
        "8": "Nord-Ouest", "9": "Ouest", "10": "Sud", "11": "Sud-Ouest", "12": "Yaoundé",
    },
}

ETHNICITE_LABELS = {
    "en": {
        "3": "Fulani", "78": "Gbaya", "101": "Tupuri", "176": "Bamum",
        "202": "Bamileke", "232": "Bassa", "235": "Eton", "236": "Ewondo",
        "245": "Bulu", "256": "Maka / Makya", "996": "Other ethnicity",
        "Autre": "Other ethnicity",
    },
    "fr": {
        "3": "Foulbé", "78": "Gbaya", "101": "Toupouri", "176": "Bamoun",
        "202": "Bamiléké", "232": "Bassa", "235": "Éton", "236": "Ewondo",
        "245": "Boulou", "256": "Maka / Makya", "996": "Autre ethnie",
        "Autre": "Autre ethnie",
    },
}

OCCUPATION_LABELS = {
    "en": {
        "11": "Farmer", "33": "Teacher",
        "62": "Trader / vendor", "66": "Vehicle driver",
        "72": "Construction trades", "73": "Metalwork / mechanics",
        "82": "Police / military", "96": "Other occupation",
        "Autre": "Other occupation",
        "Manquant": "No current partner / not reported",
    },
    "fr": {
        "11": "Agriculteur / cultivateur", "33": "Enseignant",
        "62": "Commerçant / vendeur", "66": "Conducteur de véhicule",
        "72": "Métiers du bâtiment", "73": "Métallurgie / mécanique",
        "82": "Gendarmerie / armée", "96": "Autre métier",
        "Autre": "Autre métier",
        "Manquant": "Sans partenaire actuel / non renseigné",
    },
}

RELIGION_LABELS = {
    "en": {
        "1": "Catholic", "2": "Protestant", "3": "Other Christian", "4": "Muslim",
        "Autre": "Other / Animist / No religion",
    },
    "fr": {
        "1": "Catholique", "2": "Protestant", "3": "Autres chrétiens", "4": "Musulman",
        "Autre": "Autre / Animiste / Sans religion",
    },
}

RESIDENCE_LABELS = {
    "en": {
        "1": "Yes, he lives with her", "2": "No, he lives elsewhere", "Manquant": "Not reported",
    },
    "fr": {
        "1": "Oui, il vit avec elle", "2": "Non, il vit ailleurs", "Manquant": "Non renseigné",
    },
}

EDUCATION_LABELS = {
    "en": {0: "None", 1: "Primary school", 2: "Secondary school", 3: "Higher education"},
    "fr": {0: "Aucune", 1: "École primaire", 2: "École secondaire", 3: "Études supérieures"},
}

LABELS_PAR_VARIABLE = {
    "en": {
        "region": REGION_LABELS["en"],
        "ethnicite": ETHNICITE_LABELS["en"],
        "occupation_partenaire": OCCUPATION_LABELS["en"],
        "religion": RELIGION_LABELS["en"],
        "residence_partenaire": RESIDENCE_LABELS["en"],
    },
    "fr": {
        "region": REGION_LABELS["fr"],
        "ethnicite": ETHNICITE_LABELS["fr"],
        "occupation_partenaire": OCCUPATION_LABELS["fr"],
        "religion": RELIGION_LABELS["fr"],
        "residence_partenaire": RESIDENCE_LABELS["fr"],
    },
}

OTHER_CODE_FALLBACK = {"en": "Other ({code})", "fr": "Autre ({code})"}


def libelle(colonne, code):
    trouve = LABELS_PAR_VARIABLE[lang].get(colonne, {}).get(code)
    if trouve is not None:
        return trouve
    return OTHER_CODE_FALLBACK[lang].format(code=code)


# ---------------------------------------------------------------------------
# MODEL VARIABLE LABELS, USED FOR DISPLAY IN THE SHAP CHART (per language)
# ---------------------------------------------------------------------------
NOMS_VARIABLES = {
    "en": {
        "ctrl_score": "Partner's control over her activities",
        "transmission_intergen": "Violence experienced by the woman's mother (childhood)",
        "alcool_partenaire": "Partner's alcohol consumption",
        "ecart_age": "Age gap in the couple",
        "age_femme": "Woman's age",
        "age_partenaire": "Partner's age",
        "age_premiere_union": "Age at first union",
        "age_premier_rapport": "Age at first sexual intercourse",
        "richesse": "Household wealth level",
        "education_partenaire": "Partner's education level",
        "education_femme": "Woman's education level",
        "attitude_score": "Attitude towards intimate partner violence",
        "autonomie_score": "Woman's decision-making autonomy",
        "polygamie": "Partner's polygamy",
        "milieu": "Area of residence",
        "ethnicite": "Ethnicity",
        "religion": "Religion",
        "region": "Region",
        "occupation_partenaire": "Partner's occupation",
        "residence_partenaire": "Cohabitation with partner",
    },
    "fr": {
        "ctrl_score": "Contrôle du partenaire sur ses activités",
        "transmission_intergen": "Violence subie par la mère de la femme (enfance)",
        "alcool_partenaire": "Consommation d'alcool du partenaire",
        "ecart_age": "Écart d'âge dans le couple",
        "age_femme": "Âge de la femme",
        "age_partenaire": "Âge du partenaire",
        "age_premiere_union": "Âge lors de la première mise en couple",
        "age_premier_rapport": "Âge lors du premier rapport intime",
        "richesse": "Niveau de vie du foyer",
        "education_partenaire": "Niveau d'études du partenaire",
        "education_femme": "Niveau d'études de la femme",
        "attitude_score": "Attitude envers la violence conjugale",
        "autonomie_score": "Autonomie décisionnelle de la femme",
        "polygamie": "Polygamie du partenaire",
        "milieu": "Milieu de résidence",
        "ethnicite": "Ethnie",
        "religion": "Religion",
        "region": "Région",
        "occupation_partenaire": "Métier du partenaire",
        "residence_partenaire": "Cohabitation avec le partenaire",
    },
}

MISSING_DATA_SUFFIX = {"en": " (missing data)", "fr": " (donnée manquante)"}
SHAP_LABEL_SEPARATOR = {"en": ": ", "fr": " : "}


def nom_lisible(colonne: str) -> str:
    """Translate a technical model column name into a readable label for the SHAP chart."""
    noms = NOMS_VARIABLES[lang]
    if colonne in noms:
        return noms[colonne]
    if colonne.endswith("_manquant"):
        base = colonne[: -len("_manquant")]
        return f"{noms.get(base, base)}{MISSING_DATA_SUFFIX[lang]}"
    for var_cat in CAT_VARS:
        prefixe = f"{var_cat}_"
        if colonne.startswith(prefixe):
            code = colonne[len(prefixe):]
            return f"{noms.get(var_cat, var_cat)}{SHAP_LABEL_SEPARATOR[lang]}{libelle(var_cat, code)}"
    return colonne


def _index_slider(question: str, options: list) -> int:
    """Text-choice slider: returns the index of the chosen option (used by the model)."""
    choix = st.select_slider(question, options=options, value=options[0])
    return options.index(choix)


# ---------------------------------------------------------------------------
# ICONS (inline SVG, no emoji) FOR THE CUSTOM HTML
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
# UI TEXT, per language
# ---------------------------------------------------------------------------
TEXTS = {
    "en": {
        "hero_title": "Does my situation present a risk?",
        "hero_subtitle": (
            "Answer a few simple questions to get an indicative estimate of the "
            "level of risk of physical violence in a couple."
        ),
        "tab_about": ":material/info: About",
        "tab_predict": ":material/search: Get an estimate",
        "about_purpose_title": "The purpose of this tool",
        "about_purpose_text": (
            "This application helps you reflect on a couple's situation. "
            "By answering a few simple questions, you get an estimate of "
            "the level of risk of physical violence — <b>low</b>, "
            "<b>moderate</b>, or <b>high</b>."
        ),
        "about_created_title": "Created by",
        "about_know_title": "What you should know",
        "about_know_text": (
            "The result shown is an estimate based on situations already "
            "observed in the past. <b>This is not a diagnosis</b> and it "
            "does not replace the advice of a doctor, a social worker, or "
            "any other qualified professional. If you are in danger, "
            "contact the local authorities or a victim support organization."
        ),
        "section_woman": "The woman",
        "section_partner": "The partner",
        "section_family": "The couple and family",
        "q_age_femme": "Her age",
        "q_age_premiere_union": "Her age at her first union",
        "q_age_premier_rapport": "Her age at her first sexual intercourse",
        "q_education_femme": "Her education level",
        "q_richesse": "Her household's wealth level",
        "richesse_opts": {1: "Very poor", 2: "Poor", 3: "Average", 4: "Comfortable", 5: "Wealthy"},
        "q_milieu": "Where does she live?",
        "milieu_opts": ("Urban area", "Rural area"),
        "q_age_partenaire": "His age",
        "q_education_partenaire": "His education level",
        "q_alcool": "Does he drink alcohol?",
        "alcool_opts": {0: "Never", 1: "Often", 2: "Sometimes"},
        "q_ctrl": "Does he control her activities (outings, money, friends)?",
        "ctrl_opts": ["Not at all", "A little", "Moderately", "Quite a lot", "A lot", "Extremely"],
        "q_polygamie": "Does he have other wives?",
        "polygamie_opts": ["None", "1 other", "2 others", "3 others"],
        "q_transmission": "Did the woman's father beat her mother when she was a child?",
        "transmission_opts": ("No", "Yes"),
        "q_attitude": "Does she think that hitting one's wife can sometimes be justified?",
        "attitude_opts": ["Never", "Rarely", "Sometimes", "Often", "Most of the time", "Always"],
        "q_autonomie": "Does she make her own decisions about money, her health, and going out?",
        "autonomie_opts": ["Rarely", "Sometimes", "Often", "Always"],
        "q_region": "Region where the couple lives",
        "q_ethnicite": "Woman's ethnicity",
        "q_religion": "Woman's religion",
        "q_occupation": "Partner's occupation",
        "q_residence": "Does the partner currently live with her?",
        "submit_button": "See the result",
        "result_title": "Result",
        "result_metric_label": "Estimated likelihood",
        "result_risk_level_label": "Risk level:",
        "risk_labels": {"low": "Low", "medium": "Moderate", "high": "High"},
        "result_caption": (
            "This result is an indicative estimate. It does not replace the "
            "advice of a professional (doctor, social worker, support organization)."
        ),
        "shap_title": "Factors explaining this estimate",
        "shap_caption": (
            "Each bar represents a factor's contribution to the risk score "
            "(log-odds scale, before conversion to a percentage): in red, the "
            "factors that increase the estimated risk; in blue, those that "
            "decrease it. This breakdown, based on Shapley values (SHAP "
            "method), is provided as an indication to help understand the "
            "estimate above — it is not a causal explanation."
        ),
        "footer_text": (
            "This tool is meant to support reflection — it does not replace "
            "professional medical or social advice."
        ),
        "error_model_missing": (
            "Model file not found ({path}). Run `python train_model.py` first "
            "to train and save the model."
        ),
    },
    "fr": {
        "hero_title": "Est-ce que ma situation présente un risque ?",
        "hero_subtitle": (
            "Répondez à quelques questions simples pour obtenir une estimation, à titre "
            "indicatif, du niveau de risque de violence physique dans un couple."
        ),
        "tab_about": ":material/info: À propos",
        "tab_predict": ":material/search: Faire une estimation",
        "about_purpose_title": "Le but de cet outil",
        "about_purpose_text": (
            "Cette application aide à réfléchir sur une situation de couple. "
            "En répondant à quelques questions simples, vous obtenez une estimation "
            "du niveau de risque de violence physique — <b>faible</b>, <b>modéré</b> "
            "ou <b>élevé</b>."
        ),
        "about_created_title": "Réalisé par",
        "about_know_title": "Ce qu'il faut savoir",
        "about_know_text": (
            "Le résultat affiché est une estimation basée sur des situations déjà "
            "observées par le passé. <b>Ce n'est pas un diagnostic</b> et cela ne "
            "remplace pas l'avis d'un médecin, d'un travailleur social ou de toute "
            "autre personne qualifiée. En cas de danger, rapprochez-vous des "
            "autorités locales ou d'une association d'aide aux victimes."
        ),
        "section_woman": "La femme",
        "section_partner": "Le partenaire",
        "section_family": "Le couple et la famille",
        "q_age_femme": "Son âge",
        "q_age_premiere_union": "Son âge lors de sa première mise en couple",
        "q_age_premier_rapport": "Son âge lors de son premier rapport intime",
        "q_education_femme": "Son niveau d'études",
        "q_richesse": "Le niveau de vie de son foyer",
        "richesse_opts": {1: "Très modeste", 2: "Modeste", 3: "Moyen", 4: "Aisé", 5: "Riche"},
        "q_milieu": "Où vit-elle ?",
        "milieu_opts": ("En ville", "À la campagne"),
        "q_age_partenaire": "Son âge",
        "q_education_partenaire": "Son niveau d'études",
        "q_alcool": "Boit-il de l'alcool ?",
        "alcool_opts": {0: "Jamais", 1: "Souvent", 2: "Parfois"},
        "q_ctrl": "Contrôle-t-il ses activités (sorties, argent, amis) ?",
        "ctrl_opts": ["Pas du tout", "Un peu", "Modérément", "Assez", "Beaucoup", "Énormément"],
        "q_polygamie": "A-t-il d'autres épouses ?",
        "polygamie_opts": ["Aucune", "1 autre", "2 autres", "3 autres"],
        "q_transmission": "Le père de la femme battait-il sa mère quand elle était enfant ?",
        "transmission_opts": ("Non", "Oui"),
        "q_attitude": "Pense-t-elle que frapper sa femme peut parfois être justifié ?",
        "attitude_opts": ["Jamais", "Rarement", "Parfois", "Souvent", "La plupart du temps", "Toujours"],
        "q_autonomie": "Décide-t-elle elle-même pour l'argent, sa santé et ses sorties ?",
        "autonomie_opts": ["Rarement", "Parfois", "Souvent", "Toujours"],
        "q_region": "Région où vit le couple",
        "q_ethnicite": "Ethnie de la femme",
        "q_religion": "Religion de la femme",
        "q_occupation": "Métier du partenaire",
        "q_residence": "Le partenaire vit-il avec elle actuellement ?",
        "submit_button": "Voir le résultat",
        "result_title": "Résultat",
        "result_metric_label": "Chances estimées",
        "result_risk_level_label": "Niveau de risque :",
        "risk_labels": {"low": "Faible", "medium": "Modéré", "high": "Élevé"},
        "result_caption": (
            "Ce résultat est une estimation à titre indicatif. Il ne remplace pas "
            "l'avis d'un professionnel (médecin, travailleur social, association d'aide)."
        ),
        "shap_title": "Facteurs qui expliquent cette estimation",
        "shap_caption": (
            "Chaque barre représente la contribution d'un facteur au score de risque "
            "(échelle log-odds, avant transformation en pourcentage) : en rouge, les "
            "facteurs qui augmentent le risque estimé ; en bleu, ceux qui le diminuent. "
            "Cette décomposition, basée sur les valeurs de Shapley (méthode SHAP), est "
            "fournie à titre indicatif pour mieux comprendre l'estimation ci-dessus — ce "
            "n'est pas une explication causale."
        ),
        "footer_text": (
            "Cet outil aide à la réflexion — il ne remplace pas un avis médical ou "
            "social professionnel."
        ),
        "error_model_missing": (
            "Fichier modèle introuvable ({path}). Lancez d'abord `python train_model.py` "
            "pour entraîner et sauvegarder le modèle."
        ),
    },
}

txt = TEXTS[lang]

# ---------------------------------------------------------------------------
# 1. LOADING THE PRE-TRAINED MODEL (cached)
# ---------------------------------------------------------------------------
MODEL_PATH = "model.joblib"

CAT_VARS = ["ethnicite", "religion", "region", "occupation_partenaire", "residence_partenaire"]
CIBLE = "violence_physique"


@st.cache_resource
def charger_modele():
    if not os.path.exists(MODEL_PATH):
        st.error(txt["error_model_missing"].format(path=MODEL_PATH))
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


@st.cache_resource
def charger_fond_shap():
    """Recomputes the encoded training set, used as the SHAP reference (background) data."""
    bundle_entrainement = train_model.entrainer_modele()
    return bundle_entrainement["X_entrainement"]


@st.cache_resource
def construire_explainer(_model, _fond):
    return shap.LinearExplainer(_model, _fond)


X_fond_shap = charger_fond_shap().reindex(columns=colonnes_entrainement, fill_value=0)
explainer_shap = construire_explainer(model, X_fond_shap)

# ---------------------------------------------------------------------------
# 2. HEADER
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero">
        <span class="brand-pill">{icon("shield")} SocaStat</span>
        <h1>{txt["hero_title"]}</h1>
        <p>{txt["hero_subtitle"]}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# 3. TABS: ABOUT / PREDICTION
# ---------------------------------------------------------------------------
tab_apropos, tab_prediction = st.tabs([txt["tab_about"], txt["tab_predict"]])

with tab_apropos:
    st.markdown(
        f"""
        <div class="about-card">
            <h3>{icon("flag")} {txt["about_purpose_title"]}</h3>
            <p>{txt["about_purpose_text"]}</p>
            <hr class="about-divider">
            <h3>{icon("users")} {txt["about_created_title"]}</h3>
            <div>
                <span class="author-chip">Telesphore Ebanga-Mballa</span>
                <span class="author-chip">Stéphane Bella-Mbarga</span>
                <span class="author-chip">Brenda Enow</span>
                <span class="author-chip">Kum-Collins</span>
                <span class="author-chip">Georges Nguefack-Tsague</span>
            </div>
            <hr class="about-divider">
            <h3>{icon("warning")} {txt["about_know_title"]}</h3>
            <p>{txt["about_know_text"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with tab_prediction:
    # -----------------------------------------------------------------------
    # 3.1 INPUT FORM
    # -----------------------------------------------------------------------
    with st.form("formulaire_prediction"):
        col1, col2, col3 = st.columns(3)

        with col1:
            with st.container(border=True):
                st.markdown(f'<div class="section-title">{icon("user")} {txt["section_woman"]}</div>', unsafe_allow_html=True)
                age_femme = st.slider(txt["q_age_femme"], 15, 49, 28)
                age_premiere_union = st.slider(txt["q_age_premiere_union"], 10, 40, 18)
                age_premier_rapport = st.slider(txt["q_age_premier_rapport"], 10, 40, 17)
                education_femme = st.selectbox(
                    txt["q_education_femme"], [0, 1, 2, 3], index=1,
                    format_func=lambda x: EDUCATION_LABELS[lang][x],
                    key="education_femme",
                )
                richesse = st.select_slider(
                    txt["q_richesse"],
                    options=[1, 2, 3, 4, 5], value=3,
                    format_func=lambda x: txt["richesse_opts"][x],
                )
                milieu = st.radio(
                    txt["q_milieu"], [0, 1],
                    format_func=lambda x: txt["milieu_opts"][x], horizontal=True
                )

        with col2:
            with st.container(border=True):
                st.markdown(f'<div class="section-title">{icon("user")} {txt["section_partner"]}</div>', unsafe_allow_html=True)
                age_partenaire = st.slider(txt["q_age_partenaire"], 15, 90, 35)
                education_partenaire = st.selectbox(
                    txt["q_education_partenaire"], [0, 1, 2, 3], index=1,
                    format_func=lambda x: EDUCATION_LABELS[lang][x],
                    key="education_partenaire",
                )
                alcool_partenaire = st.selectbox(
                    txt["q_alcool"], [0, 1, 2],
                    format_func=lambda x: txt["alcool_opts"][x]
                )
                ctrl_score = _index_slider(txt["q_ctrl"], txt["ctrl_opts"])
                polygamie = _index_slider(txt["q_polygamie"], txt["polygamie_opts"])

        with col3:
            with st.container(border=True):
                st.markdown(f'<div class="section-title">{icon("home")} {txt["section_family"]}</div>', unsafe_allow_html=True)
                transmission_intergen = st.radio(
                    txt["q_transmission"], [0, 1],
                    format_func=lambda x: txt["transmission_opts"][x], horizontal=True
                )
                attitude_score = _index_slider(txt["q_attitude"], txt["attitude_opts"])
                autonomie_score = _index_slider(txt["q_autonomie"], txt["autonomie_opts"])
                region = st.selectbox(
                    txt["q_region"], sorted(categories_valides["region"], key=lambda c: int(c)),
                    format_func=lambda c: libelle("region", c)
                )
                ethnicite = st.selectbox(
                    txt["q_ethnicite"], categories_valides["ethnicite"],
                    format_func=lambda c: libelle("ethnicite", c)
                )
                religion = st.selectbox(
                    txt["q_religion"], categories_valides["religion"],
                    format_func=lambda c: libelle("religion", c)
                )
                occupation_partenaire = st.selectbox(
                    txt["q_occupation"], categories_valides["occupation_partenaire"],
                    format_func=lambda c: libelle("occupation_partenaire", c)
                )
                residence_partenaire = st.selectbox(
                    txt["q_residence"], categories_valides["residence_partenaire"],
                    format_func=lambda c: libelle("residence_partenaire", c)
                )

        st.markdown("<br>", unsafe_allow_html=True)
        valider = st.form_submit_button(
            txt["submit_button"], icon=":material/search:", type="primary", use_container_width=True
        )

    # -----------------------------------------------------------------------
    # 3.2 BUILDING THE OBSERVATION AND PREDICTING
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

        # missing-value indicators (none here, the form is fully filled -> all set to 0)
        for col in num_vars:
            col_manquant = f"{col}_manquant"
            if col_manquant in colonnes_entrainement:
                df_obs[col_manquant] = 0

        # one-hot encoding of categorical variables, aligned on the training columns
        df_obs_encoded = pd.get_dummies(df_obs, columns=CAT_VARS, prefix=CAT_VARS)
        df_obs_encoded = df_obs_encoded.reindex(columns=colonnes_entrainement, fill_value=0)

        proba = model.predict_proba(df_obs_encoded)[0, 1]
        proba_pct = min(proba, 1.0) * 100

        if proba < 0.25:
            niveau, classe = txt["risk_labels"]["low"], "low"
        elif proba < 0.50:
            niveau, classe = txt["risk_labels"]["medium"], "medium"
        else:
            niveau, classe = txt["risk_labels"]["high"], "high"

        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="section-title">{icon("check")} {txt["result_title"]}</div>', unsafe_allow_html=True)

        c1, c2 = st.columns([1, 2])

        with c1:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="value">{proba:.0%}</div>
                    <div class="label">{txt["result_metric_label"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c2:
            st.markdown(
                f"""
                <p>{txt["result_risk_level_label"]} <span class="badge badge-{classe}">{niveau}</span></p>
                <div class="gauge-track">
                    <div class="gauge-fill {classe}" style="width: {proba_pct:.1f}%;"></div>
                </div>
                <div class="gauge-caption">
                    <span>{txt["risk_labels"]["low"]}</span>
                    <span>{txt["risk_labels"]["medium"]}</span>
                    <span>{txt["risk_labels"]["high"]}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

        st.caption(txt["result_caption"])

        # ---------------------------------------------------------------
        # 3.3 EXPLAINING THE PREDICTION (SHAPLEY / SHAP VALUES)
        # ---------------------------------------------------------------
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f'<div class="section-title">{icon("flag")} {txt["shap_title"]}</div>',
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            valeurs_shap = explainer_shap(df_obs_encoded)
            explication = shap.Explanation(
                values=valeurs_shap.values[0],
                base_values=valeurs_shap.base_values[0],
                data=df_obs_encoded.iloc[0].values,
                feature_names=[nom_lisible(c) for c in colonnes_entrainement],
            )

            fig, ax = plt.subplots()
            shap.plots.waterfall(explication, max_display=12, show=False)
            plt.tight_layout()
            st.pyplot(fig, clear_figure=True)

            st.caption(txt["shap_caption"])

# ---------------------------------------------------------------------------
# 4. FOOTER
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="app-footer">
        {txt["footer_text"]}
    </div>
    """,
    unsafe_allow_html=True,
)
