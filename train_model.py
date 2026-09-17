"""
Entraine le modele de prediction du risque de violence physique conjugale
et sauvegarde le modele + les artefacts de pretraitement sur disque.

A lancer une seule fois (ou chaque fois que le dataset change) :

    python train_model.py

L'application Streamlit (app_streamlit.py) charge ensuite le fichier
produit ici (model.joblib) au lieu de re-entrainer le modele a chaque
demarrage.
"""
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

DATA_PATH = "dataset_violence.csv"
MODEL_PATH = "model.joblib"

CAT_VARS = ["ethnicite", "religion", "region", "occupation_partenaire", "residence_partenaire"]
CIBLE = "violence_physique"
SEUIL_RARE = 0.02


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

    return {
        "model": model,
        "colonnes_entrainement": X.columns.tolist(),
        "medianes": medianes,
        "categories_valides": categories_valides,
        "num_vars": num_vars,
    }


if __name__ == "__main__":
    bundle = entrainer_modele()
    joblib.dump(bundle, MODEL_PATH)
    print(f"Modele entraine sur {DATA_PATH} et sauvegarde dans {MODEL_PATH}")
