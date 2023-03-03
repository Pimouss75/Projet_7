# Importation des bibliothèques nécessaires
import joblib
import dill
import uvicorn
import numpy as np
import pandas as pd
from pydantic import BaseModel

# Importation de la bibliothèque FastAPI
from fastapi import FastAPI, HTTPException

# Initiate app instance
app = FastAPI(title='Admissibilité à un prêt', version='1.0',
              description='Modèle optimisé pour la prédiction')

# Chargement des fichiers d'artefacts de modèle au lancement du serveur FastAPI
model = joblib.load("model.pkl")
df = pd.read_csv("df_test.csv")


# Définition d'une classe Pydantic pour valider la requête
class PredictionData(BaseModel):
    SK_ID_CURR: str

# Point de terminaison racine ou de base
@app.get('/')
@app.get('/home')
def read_home():
    """
     Endpoint de base qui peut être utilisé pour tester la disponibilité de l'application.
     """
    return {'message': 'Le système est opérationnel'}

# Point de terminaison pour effectuer des prédictions en fonction de l'ID du client
@app.post("/predict/{client_id}")
def predict(prediction_data: PredictionData, client_id: str):
    df.SK_ID_CURR = df.SK_ID_CURR.astype('str')
    client_id = str(client_id)
    data = df[df["SK_ID_CURR"] == client_id].drop(['TARGET', 'SK_ID_CURR'], axis=1)

    if len(data) == 0:
        prediction_result = "Numero client invalide."
        return prediction_result
    else:
        # Seuil optimal
        seuil = 0.50
        prediction = model.predict_proba(data)
        if prediction[0][1] > seuil:
            prediction_result = "Client susceptible d'avoir des difficultés à rembourser un prêt."
        else:
            prediction_result = "Client à faible risque d'être en défaut de paiement si un prêt lui est consenti."
        return prediction_result


# Point de terminaison pour obtenir les données d'un client en fonction de son ID
@app.get("/client/{client_id}")
def get_client(client_id: str) :
    df.SK_ID_CURR = df.SK_ID_CURR.astype('str')
    client_id = str(client_id)
    data = df[df["SK_ID_CURR"] == client_id].drop(['TARGET', 'SK_ID_CURR'], axis=1)

    if len(data) == 0:
        raise HTTPException(status_code=404, detail=f"Client {client_id} non référencé.")
    else:
        data = data.to_dict(orient="records")[0]
        return {"data": data}


@app.get("/all_X_test")
def all_X_test() :
    data1 = df.drop(['TARGET', 'SK_ID_CURR'], axis=1)
    # apply the preprocessing to x_test
#    data3 = model['preprocessors'].transform(data1)
#    data3 = pd.DataFrame(data3, index=data1.index, columns=data1.columns)
#    del data1
    data2 = data1.to_dict(orient="records")[0]
#    del data3
    return {"data2": data2}

if __name__ == '__main__':
    uvicorn.run("main:app", host="localhost", port=80, reload=True)

# lancer l'api: uvicorn main:app --reload