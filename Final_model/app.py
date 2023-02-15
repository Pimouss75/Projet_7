import streamlit
import requests as re
import json
import pandas as pd

df = pd.read_csv("df_test.csv")



def run():
    streamlit.title("Prediction d'admissibilité a un pret")
    client_id = streamlit.number_input("Reference client SK_ID_CURR : ")


    data_pred = {
        "Reference client SK_ID_CURR : ":client_id,
    }


    if streamlit.button("Predict"):
        streamlit.info(data_pred)
        #response = requests.post("http://0.0.0.0:8000/predict", data=json.dumps(data_pred, default=str))
        #response = requests.post("http://127.0.0.1:8000/predict", json=data_pred )
        #prediction = response.text
        #streamlit.success(f"The prediction from model: {prediction}")
        response = re.post("http://127.0.0.1:8000/predict", data=json.dumps(data_pred, default=str))
        prediction = response.text
        streamlit.success(f"The prediction from model: {prediction}")



if __name__ == '__main__':
    # by default it will run at 8501 port
    run()

# lancer l'app streamlit run app.py