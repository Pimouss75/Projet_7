import streamlit as st
import requests
import json
import pandas as pd
import dill
import shap
import streamlit_shap
st.set_page_config(layout='wide')

st.set_option('deprecation.showPyplotGlobalUse', False)
url_aws = "http://127.0.0.1:8000"
button_style = """
        <style>
        .stButton > button {
            color: white;
            background: black;
            width: 150px;
            height: 80px;
                        
        }
        button{
   div{
    p{font-size:50px}
    }
        </style>
        """



st.markdown(button_style, unsafe_allow_html=True)
def run():
    st.title("Prediction d'admissibilité a un pret")
    client_id = st.text_input("Reference client SK_ID_CURR : ",max_chars = 6 )

#    data_pred = {
#        "SK_ID_CURR":client_id,
#    }

    if st.button("Predict"):
#        st.info(data_pred)
        if client_id == "":
            response = "Merci de donner un numero client"
            st.success( {response} )


        response = requests.post(f"{url_aws}/predict/" + str(client_id), data=json.dumps(client_id, default=str))
        prediction = response.text
        st.success(f"La prediction pour ce client est: \n {prediction}")

    if st.sidebar.button("Features importance"):

        # Chargement du de l'explainer
        explainer = dill.load(open("shap_explainer.dill", "rb"))

        response = requests.get(f"{url_aws}/client/" + str(client_id),verify=False)
        if response.status_code == 200:
            X_test = response.json()["data"]
#            X_test = pd.DataFrame(X_test)
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