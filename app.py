import streamlit as st
import requests
import json
import pandas as pd
import dill
import shap
import streamlit_shap
import base64

st.set_page_config(layout='wide')

st.set_option('deprecation.showPyplotGlobalUse', False)

loc_aws = "http://localhost:8000"



st.title("Prédiction d'admissibilité a un prêt")

LOGO_IMAGE = "pad.png"
# st.image(LOGO_IMAGE, caption=None, width=300)
st.sidebar.image(LOGO_IMAGE,use_column_width=True)

button_style = """
        <style>
        .stButton > button {
            color: white;
            background: black;
            width: 210px;
            height: 80px;

        </style>
        """
st.markdown(button_style, unsafe_allow_html=True)

# txt = "text here"



def run():

    col, buff, buff2 = st.columns([1, 2, 2])
    client_id = col.text_input("Reference client (SK_ID_CURR) : ", max_chars=6)

    data_pred = {
            "SK_ID_CURR":client_id,
        }

    col1, col2, col3 = st.columns([20, 20, 50])

    def la_rep():

        if client_id == "":
            response = "Merci de fournir une référence client"
            st.write(response)

        else :
            response = requests.post(f"{loc_aws}/predict/" + str(client_id), data=json.dumps(data_pred, default=str))
            prediction = response.text
            st.success(f"La prediction pour ce client est: \n {prediction}")


    if col1.button("Prediction"):
#       st.info(data_pred)
        la_rep()

    if col2.button("Features importance"):
        la_rep()

        # Chargement du de l'explainer
        explainer = dill.load(open("shap_explainer.dill", "rb"))

        response = requests.get(f"{loc_aws}/client/" + str(client_id), verify=False)
        if response.status_code == 200:
            X_test = response.json()["data"]
            # X_test = pd.DataFrame(X_test)
            X_test = pd.DataFrame(X_test, index=pd.RangeIndex(len(X_test)))

            shap_values = explainer.shap_values(X_test)

            st.subheader("Interprétabilité shap du client")
            # Affichage du graphique

            streamlit_shap.st_shap(shap.force_plot(explainer.expected_value[0],
                                                   shap_values[0][1],
                                                   feature_names=X_test.columns))

            shap.plots._waterfall.waterfall_legacy(explainer.expected_value[0],
                                                   shap_values[0][1],
                                                   feature_names=X_test.columns,
                                                   max_display=10)
            st.pyplot()


if __name__ == '__main__':
    # by default it will run at 8501 port
    run()

# lancer l'app streamlit run app.py