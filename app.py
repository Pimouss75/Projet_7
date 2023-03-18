import streamlit as st
import requests
import json
import pandas as pd
import dill
import shap
import streamlit_shap
import numpy as np
from streamlit_plotly_events import plotly_events
import plotly.express as px
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

        response2 = requests.get(f"{loc_aws}/client2/{client_id}" , verify=False)
        if response2.status_code == 200:
            st.subheader(f"Informations relatives au client {client_id}")
            X_test = response2.json()["data"]
            X_test = pd.DataFrame(X_test)
            info_df = pd.concat([pd.DataFrame(descr_var).set_index('Information'),
                       X_test.T], axis=1, join="inner")
            info_df = info_df.rename(columns={0 : 'Valeur'})
            info_df = info_df.rename(
                index={'DAYS_BIRTH': 'Age', 'CODE_GENDER': 'SEXE', 'DAYS_EMPLOYED': 'YEARS_EMPLOYED'})
#            columns = info_df.index.tolist()
            st.dataframe(info_df)

        else:
            st.write("echec de la requete", response2.status_code)


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



    st.sidebar.header("Choix de la variable à afficher")
    # liste déroulante variables

    nom_colonnes = ['REG_CITY_NOT_WORK_CITY', 'REGION_RATING_CLIENT', 'DAYS_ID_PUBLISH',
                     'Age', 'SEXE', 'REG_CITY_NOT_LIVE_CITY', 'DAYS_LAST_PHONE_CHANGE',
                    'EXT_SOURCE_2', 'YEARS_EMPLOYED', 'EXT_SOURCE_3']


    column_sel = nom_colonnes[0]
    nom_colonnes = pd.Series(nom_colonnes)

    column_sel = st.sidebar.selectbox('Sélectionnez la variable :', options=nom_colonnes)
    # les donnes des clients du dataset de cette colonne

    response4 = requests.get(f"{loc_aws}/col_choix/{column_sel}", verify=False)
    response2 = requests.get(f"{loc_aws}/client2/{client_id}" , verify=False)

#    val_cli = None
    if response2.status_code == 200 and response4.status_code == 200:
        X_test2 = response2.json()["data"]
        X_test2 = pd.DataFrame(X_test2)
        X_test2 = X_test2.replace('?', np.nan)
        X_test2['CODE_GENDER'] = X_test2['CODE_GENDER'].apply(lambda x: 1 if x == 'F' else 0)
        X_test2 = X_test2.rename(columns={'DAYS_BIRTH': 'Age', 'DAYS_EMPLOYED': 'YEARS_EMPLOYED',
                                                'CODE_GENDER': 'SEXE'})

        if np.isnan(X_test2[column_sel]).any():
            data_col = response4.json()["data3"]
            data_col = pd.DataFrame(data_col)
            data_col = data_col.replace('?', np.nan)
            # Display the figure
            st.subheader("Histogramme de tous les clients de test :")
            fig = px.histogram(data_col, x=column_sel)
            fig.write_html("test.html")
            selected_points = plotly_events(fig)
            st.write(selected_points)

        else:
            val_cli = round(X_test2[column_sel].values[0], 1)
            data_col = response4.json()["data3"]
            data_col = pd.DataFrame(data_col)
            data_col = data_col.replace('?', np.nan)
            # Display the figure
            st.subheader("Position du client par rapport aux autres :")
            fig = px.histogram(data_col, x=column_sel)
            fig.add_vline(x=val_cli, line_width=3, line_dash="dash", line_color="green")
            fig.write_html("test.html")
            selected_points = plotly_events(fig)
            st.write(selected_points)

    else :
        data_col = response4.json()["data3"]
        data_col = pd.DataFrame(data_col)
        data_col = data_col.replace('?', np.nan)
        # Display the figure
        st.subheader("Histogramme de tous les clients de test :")
        fig = px.histogram(data_col, x=column_sel)
        fig.write_html("test.html")
        selected_points = plotly_events(fig)
        st.write(selected_points)

if __name__ == '__main__':
    # by default it will run at 8501 port
    run()

# lancer l'app streamlit run app.py