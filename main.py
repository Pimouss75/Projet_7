# Import Needed Libraries
import joblib
import dill
import uvicorn
import numpy as np
import pandas as pd
from pydantic import BaseModel

# FastAPI libray
from fastapi import FastAPI

# Initiate app instance
app = FastAPI(title='Admissibilité a un pret', version='1.0',
              description='model optimise pour la prediction')

# Initialize model artifact files. This will be loaded at the start of FastAPI model server.
model = joblib.load("model.pkl")
explainer = dill.load(open("shap_explainer.dill","rb"))
df = pd.read_csv("df_test.csv")



# This struture will be used for Json validation.
# With just that Python type declaration, FastAPI will perform below operations on the request data
## 1) Read the body of the request as JSON.
## 2) Convert the corresponding types (if needed).
## 3) Validate the data.If the data is invalid, it will return a nice and clear error,
##    indicating exactly where and what was the incorrect data.

class Data(BaseModel):
    SK_ID_CURR: str



# Api root or home endpoint
@app.get('/')
@app.get('/home')
def read_home():
    """
     Home endpoint which can be used to test the availability of the application.
     """
    return {'message': 'System is healthy'}


# ML API endpoint for making prediction aganist the request received from client
@app.post("/predict/{client_id}")
def predict(client_id:str):
    df.SK_ID_CURR = df.SK_ID_CURR.astype('str')
    client_id = str(client_id)
    data = df[df["SK_ID_CURR"] == client_id].drop(['TARGET', 'SK_ID_CURR'], axis=1)
    if len(data) == 0 :
        prediction_result = f"Client {client_id} non référencé, prediction impossible."

    else:
#        data = data.iloc[0].to_dict()
#        prediction = model.predict_proba(pd.DataFrame(columns=list(data.keys()), data=[list(data.values())]))
        prediction = model.predict_proba(data)

        if prediction[0][1] > 0.54:
            prediction_result = "Client susceptible d'avoir des difficultés à rembourser un prêt."
#            shap_values = explainer.shap_values(data)
#            shap = shap.force_plot(explainer.expected_value[0], shap_values[0], feature_names=data.columns)
        else:
            prediction_result = "Client à faible risque d'être en défaut de paiement si un prêt lui est consenti."
#            shap_values = explainer.shap_values(data)
#            shap = shap.force_plot(explainer.expected_value[0], shap_values[1], feature_names=data.columns)


    return prediction_result


if __name__ == '__main__':
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

# lancer l'api: uvicorn main:app --reload