import streamlit as st
import requests
import json
import pandas as pd
import dill
import shap
import streamlit_shap
import base64

st.set_page_config(layout='wide',page_title='Prêt à dépenser')
st.set_option('deprecation.showPyplotGlobalUse', False)
loc_aws = "http://localhost:8000"

st.title("Prédiction d'admissibilité a un prêt")

# image interpretation globale
gl_test = "app_test_global.png"

# logo societe
LOGO_IMAGE = "pad.png"
# st.image(LOGO_IMAGE, caption=None, width=300)
st.sidebar.image(LOGO_IMAGE,use_column_width=True)

button_style = """
        <style>
        .stButton > button {
            color: white;
            background: black;
            width: 200px;
            height: 85px;

        </style>
        """
st.markdown(button_style, unsafe_allow_html=True)

descr_var = {

    'Information' : ['REG_CITY_NOT_WORK_CITY','REGION_RATING_CLIENT','DAYS_ID_PUBLISH',
                     'DAYS_BIRTH','CODE_GENDER','REG_CITY_NOT_LIVE_CITY','DAYS_LAST_PHONE_CHANGE',
                     'EXT_SOURCE_2','DAYS_EMPLOYED','EXT_SOURCE_3'],

    'Description' : ["Flag if client's permanent address does not match work address (1=different, 0=same, at city level)",
                     "Our rating of the region where client lives (1,2,3)",
                     "How many days before the application did client change the identity document with which he applied for the loan",
                     "Client's age in years at the time of application","Gender of the client",
                     "Flag if client's permanent address does not match contact address (1=different, 0=same, at city level)",
                     "How many days before application did client change phone",
                     "Normalized score from external data source",
                     "How many years before the application the person started current employment",
                     "Normalized score from external data source"],
    }


def run():

    col, buff, buff2 = st.columns([1, 2, 2])
    client_id = col.text_input("Reference client (SK_ID_CURR) : ", max_chars=6)

    data_pred = {
            "SK_ID_CURR":client_id,
        }

    col1, col2, col3 = st.columns([20, 20, 20])

    def la_rep():

        if client_id == "":
            response = "Merci de fournir une référence client"
            st.write(response)

        else :
            response = requests.post(f"{loc_aws}/predict/" + str(client_id), data=json.dumps(data_pred, default=str))
            prediction = response.text
            st.success(f"La prediction pour ce client est: \n {prediction}")


    if col1.button("Prediction"):
        la_rep()

        response2 = requests.get(f"{loc_aws}/client2/" + str(client_id), verify=False)
        if response2.status_code == 200:
            st.subheader(f"Informations relatives au client {client_id}")
            X_test = response2.json()["data"]
            X_test = pd.DataFrame(X_test)
#            X_test = pd.DataFrame(X_test, index=pd.RangeIndex(len(X_test))).iloc[0, :]
            X_test['DAYS_BIRTH'] = round(-X_test['DAYS_BIRTH'] / 365, 1)
            X_test['DAYS_EMPLOYED'] = round(-X_test['DAYS_EMPLOYED'] / 365, 1)
            info_df = pd.concat([pd.DataFrame(descr_var).set_index('Information'),
                       X_test.T], axis=1, join="inner")
            info_df = info_df.rename(columns={0 : 'Valeur'})
            info_df = info_df.rename(
                index={'DAYS_BIRTH': 'Age', 'CODE_GENDER': 'SEXE', 'DAYS_EMPLOYED': 'YEARS_EMPLOYED'})
            st.dataframe(info_df)


    if col3.button("Comparaison avec les autres clients"):
        la_rep()
        explainer = dill.load(open("shap_explainer.dill", "rb"))
        response2 = requests.get(f"{loc_aws}/client/" + str(client_id), verify=False)

        if response2.status_code == 200:
#            response = requests.get(f"{loc_aws}/all_X_test", verify=False)
#            X_test_all = response.json()["data2"]
#            X_test_all = pd.DataFrame(X_test_all)

            st.subheader("Interprétabilité globale")
#            shap_values = explainer.shap_values(X_test_all)
#            shap.summary_plot(shap_values, X_test_all, plot_type="bar")
#            st.pyplot()
            st.image(gl_test, output_format="auto")

            X_test = response2.json()["data"]
            X_test = pd.DataFrame(X_test, index=pd.RangeIndex(len(X_test)))
            shap_values = explainer.shap_values(X_test)

            # Affichage du graphique
            st.subheader(f"Interprétabilité locale du client {client_id}")
            streamlit_shap.st_shap(shap.force_plot(explainer.expected_value,
                                                   shap_values[0],
                                                   feature_names=X_test.columns))

            st.pyplot()

    if col2.button("Interpretation du client"):
        la_rep()

        # Chargement du de l'explainer
        explainer = dill.load(open("shap_explainer.dill", "rb"))

        response = requests.get(f"{loc_aws}/client/" + str(client_id), verify=False)
        if response.status_code == 200:
            X_test = response.json()["data"]
            X_test = pd.DataFrame(X_test)

            shap_values = explainer.shap_values(X_test)

            # Affichage du graphique
            st.subheader(f"Interprétabilité locale du client {client_id}")
            streamlit_shap.st_shap(shap.force_plot(explainer.expected_value,
                                                   shap_values[0],
                                                   feature_names=X_test.columns))


            shap.plots._waterfall.waterfall_legacy(explainer.expected_value,
                                                   shap_values[0],
                                                   feature_names=X_test.columns,
                                                   max_display=10)
            st.pyplot()



if __name__ == '__main__':
    # by default it will run at 8501 port
    run()

# lancer l'app streamlit run app.py