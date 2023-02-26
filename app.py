import streamlit
import requests
import json
import pandas as pd
import dill
import shap
import streamlit_shap

streamlit.set_option('deprecation.showPyplotGlobalUse', False)
url_aws = "http://127.0.0.1:8000"

def run():
    streamlit.title("Prediction d'admissibilité a un pret")
    client_id = streamlit.text_input("Reference client SK_ID_CURR : ")

    data_pred = {
        "SK_ID_CURR":client_id,
    }

    if streamlit.button("Predict"):
        streamlit.info(data_pred)

        response = requests.post(f"{url_aws}/predict/" + str(client_id), data=json.dumps(data_pred, default=str))
        prediction = response.text
        streamlit.success(f"La prediction pour ce client est: \n {prediction}")

    if streamlit.sidebar.button("Features importance"):

        # Chargement du de l'explainer
        explainer = dill.load(open("shap_explainer.dill", "rb"))

        response = requests.get(f"{url_aws}/client/" + str(client_id),verify=False)
        if response.status_code == 200:
            X_test = response.json()["data"]
#            X_test = pd.DataFrame(X_test)
            X_test = pd.DataFrame(X_test, index=pd.RangeIndex(len(X_test)))

            shap_values = explainer.shap_values(X_test)


            streamlit.subheader("Interprétabilité shap du client")
            # Affichage du graphique

            streamlit_shap.st_shap(shap.force_plot(explainer.expected_value[0],
                                                   shap_values[0][1],
                                                   feature_names=X_test.columns))


            shap.plots._waterfall.waterfall_legacy(explainer.expected_value[0],
                                                   shap_values[0][1],
                                                   feature_names=X_test.columns,
                                                   max_display=10)
            streamlit.pyplot()



if __name__ == '__main__':
    # by default it will run at 8501 port
    run()

# lancer l'app streamlit run app.py