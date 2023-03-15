## Projet_7
### Le projet consiste en la création d’une page internet qui affichera les résultats de prédiction d’accord ou non de crédit à pour un client d'une base de données de test. Cette prédiction sera réalisée en utilisant un algorithme capable de traiter des problèmes de classification binaire.
### Un Dashboard est créé, demande à l’utilisateur de fournir le numero de client a 6 chiffres. Le Dashboard génère une réponse à une demande de crédit avec une réponse négative ou positive, des informations relatives au client, son graphique SHAP ainsi que celui de tous les clients de la manière suivante :
#### -Lorsque la case « Prediction » aura été cliquée, la réponse à la demande de crédit s’affiche ainsi que l’explication et la valeur de chaque variable dont la description est disponible.
#### -Lorsque la case « Interpretation du client » est cliquée, la réponse à la demande de crédit s’affiche ainsi que la feature importance locale sous forme de graphique est générée en utilisant SHAP.
#### -Lorsque la case « Comparaison avec les autres clients » est cliquée, la réponse à la demande de crédit s’affiche ainsi que la feature importance locale sous forme de graphique est générée en utilisant SHAP ainsi que l’importances des variables pour tous les clients de test (globale), soit la globalité des clients (graphique des variables qui influencent tous les clients test par ordre decroissants).
### Le contenu de ce repository est composé des elements nécessaires au deploiement, dont voici le detail:
#### -P7_Marc_Sellam.ipynb : Le notebook qui a servi à l'elaboration du model de prediction qui va etre deployé.
#### -model.pkl : modele au format pickle généré par le notebook (egalement par mlflow).
#### -requirements.txt : Les librairies à deployer au prealable pour le bon fonctionement de la prediction.
#### -DOSSIER mlflow_model : Dossier généré par MLFLOW contenant des elements nécessaires au deploiement du model(requirements/model).
#### shap_explainer.dill : explicateur SHAP qui servira a generer le graphique de la feature importance du client.
#### app.py : application Dashboard (app.py) réalisée avec Streamlit qui fait appelle à l’API via une url pour récupérer la réponse à la demande de crédit, récupérer les variables et leurs valeurs respectives client afin de les décrire sous forme d’un tableau et afficher le graphique des features importances du client avec SHAP (locale) ou récupérer les variables de tous les clients de test afin d’en afficher le graphique d’importance des features sur tous les clients (globale).
#### -main.py : API (main.py) réalisée avec Fastapi, qui renverra les informations décrites ci-dessus sur demande de Streamlit (calcul de la prédiction du client, variables du clients et variables de tous les clients).
#### -df_test.csv : dataset de test composé des clients de test.
#### -pad.png : Logo "pret a depenser" utilisé lors de l'affichage du dashboard
#### -app_test_global.png : graphique des features importance de tous les clients (globale)
#### -report_data_drift.html : Rapport d’analyse au format HTML crée par la librairie Evidently ,décrit les résultats de l'analyse de dérive de données pour un ensemble de données contenant 20 caractéristiques. 
#### -DOSSIER .github/workflows : contient le fichier test_code.yml nécessaire au GitHub Actions pour un déploiement automatisé. 
#### -DOSSIER test_cases : dossier de test pytest contenant les element necessaires a son fonctionement afin que GitHub Actions effectue un déploiement automatisé si ce test est validé, soit :
#### -df_train_test.csv : dataset composé d'une dizaine  d’observations du dataset d’entrainement.
#### -model.pkl : model de prediction
#### -test_user.py : code de test pytest
#### 


# 1
## 2
### 3
#### 4
###### 5
