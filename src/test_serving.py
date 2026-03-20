"""
Test du serving MLflow — modèle LightGBM optimisé (Optuna), version 3.

Lancement préalable dans un terminal séparé :
    uv run mlflow models serve \
        -m "models:/credit-scoring-model@production" \
        --port 5001 \
        --no-conda

Puis exécuter ce script dans un autre terminal :
    uv run python src/test_serving.py
"""

import json
import requests
import pandas as pd

SEUIL_OPT = 0.4926
NB_CLIENT = 3

# Charger quelques lignes du dataset de test
df_test = pd.read_csv("data/processed/test_processed.csv")
sample = df_test.drop(columns=["SK_ID_CURR"]).head(NB_CLIENT)

# Format attendu par MLflow serving
payload = {
    "dataframe_split": {
        "columns": sample.columns.tolist(),
        "data": sample.values.tolist()
    }
}

# Appel au serveur
url = "http://127.0.0.1:5001/invocations"
headers = {"Content-Type": "application/json"}

print("Envoi de la requête au serveur MLflow...")
response = requests.post(url, data=json.dumps(payload), headers=headers)

if response.status_code == 200:
    predictions = response.json()
    print("\nRéponse du serveur MLflow :")
    print(json.dumps(predictions, indent=2))
    print("\nInterprétation :")
    for i, pred in enumerate(predictions["predictions"]):
        if isinstance(pred, list):
            # predict_proba → [proba_classe_0, proba_classe_1]
            decision = "REFUS" if pred[1] >= SEUIL_OPT else "ACCORD"
            print(f"  Client {i+1} — proba défaut : {pred[1]:.1%} → {decision}")
        else:
            # predict → 0 ou 1
            decision = "REFUS" if pred == 1 else "ACCORD"
            print(f"  Client {i+1} — classe prédite : {pred} → {decision}")
else:
    print(f"Erreur {response.status_code} : {response.text}")
