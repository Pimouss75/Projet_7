## Projet_7
### Le projet consiste en la création d’une page internet qui affichera les résultats de prédiction d’accord ou non de crédit à pour un client d'une base de données de test. Cette prédiction sera réalisée en utilisant un algorithme capable de traiter des problèmes de classification binaire.
### Un Dashboard est créé, demande à l’utilisateur de fournir le numero de client a 6 chiffres. Le Dashboard génère une réponse à une demande de crédit avec une réponse négative ou positive, des informations relatives au client, son graphique SHAP ainsi que celui de tous les clients de la manière suivante :
#### -Lorsque la case « Prediction » aura été cliquée, la réponse à la demande de crédit s’affiche ainsi que l’explication et la valeur de chaque variable dont la description est disponible.
#### -Lorsque la case « Interpretation du client » est cliquée, la réponse à la demande de crédit s’affiche ainsi que la feature importance locale sous forme de graphique est générée en utilisant SHAP.
#### -Lorsque la case « Comparaison avec les autres clients » est cliquée, la réponse à la demande de crédit s’affiche ainsi que la feature importance locale sous forme de graphique est générée en utilisant SHAP ainsi que l’importances des variables pour tous les clients de test (globale), soit la globalité des clients (graphique des variables qui influencent tous les clients test par ordre decroissants).
### Le contenu de ce repository est composé, entre autres, des éléments nécessaires au deploiement, dont voici le detail:
#### -P7_Marc_Sellam.ipynb : Le notebook qui a servi à l'elaboration du model de prediction qui va etre deployé.
#### -DOSSIER mlflow_model : Dossier généré par MLFLOW contenant des elements nécessaires au deploiement du model(requirements/model).
#### -model.pkl : modele au format pickle généré par le notebook (egalement par mlflow).
#### -requirements.txt : Les librairies à deployer au prealable pour le bon fonctionement de la prediction.
#### -shap_explainer.dill : explicateur SHAP qui servira a generer le graphique de la feature importance du client.
#### -app.py : application Dashboard (app.py) réalisée avec Streamlit qui fait appelle à l’API via une url pour récupérer la réponse à la demande de crédit, récupérer les variables et leurs valeurs respectives client afin de les décrire sous forme d’un tableau et afficher le graphique des features importances du client avec SHAP (locale) ou récupérer les variables de tous les clients de test afin d’en afficher le graphique d’importance des features sur tous les clients (globale).
#### -main.py : API (main.py) réalisée avec Fastapi, qui renverra les informations décrites ci-dessus sur demande de Streamlit (calcul de la prédiction du client, variables du clients et variables de tous les clients).
#### -df_test.csv : dataset de test composé des clients de test.
#### -pad.png : Logo "pret a depenser" utilisé lors de l'affichage du dashboard.
#### -app_test_global.png : graphique des features importance de tous les clients (globale).
#### -report_data_drift.html : Rapport d’analyse au format HTML crée par la librairie Evidently ,décrit les résultats de l'analyse de dérive de données pour un ensemble de données contenant 20 caractéristiques. 
#### -DOSSIER .github/workflows : contient le fichier test_code.yml, fichier de parametrage GitHub Actions nécessaire pour un déploiement automatisé.
#### -DOSSIER pics : captures ecrans d'images du notebook.
#### -DOSSIER test_cases : dossier de test pytest contenant les element necessaires a son fonctionement afin que GitHub Actions effectue un déploiement automatisé si ce test est validé, soit :
#### -df_train_test.csv : dataset composé d'une dizaine  d’observations du dataset d’entrainement.
#### -model.pkl : model de prediction
#### -test_user.py : code de test pytest

## Implémentez un modèle de scoring

Vous êtes Data Scientist au sein d'une société financière, nommée "Prêt à dépenser", qui propose des crédits à la consommation pour des personnes ayant peu ou pas du tout d'historique de prêt.

L’entreprise souhaite mettre en œuvre un outil de “scoring crédit” pour calculer la probabilité qu’un client rembourse son crédit, puis classifie la demande en crédit accordé ou refusé. Elle souhaite donc développer un algorithme de classification en s’appuyant sur des sources de données variées (données comportementales, données provenant d'autres institutions financières, etc.).

De plus, les chargés de relation client ont fait remonter le fait que les clients sont de plus en plus demandeurs de transparence vis-à-vis des décisions d’octroi de crédit. Cette demande de transparence des clients va tout à fait dans le sens des valeurs que l’entreprise veut incarner.

Prêt à dépenser décide donc de développer un dashboard interactif pour que les chargés de relation client puissent à la fois expliquer de façon la plus transparente possible les décisions d’octroi de crédit, mais également permettre à leurs clients de disposer de leurs informations personnelles et de les explorer facilement. 

Les données

Voici les données dont vous aurez besoin pour réaliser le dashboard. Pour plus de simplicité, vous pouvez les télécharger à cette adresse.

Vous aurez sûrement besoin de joindre les différentes tables entre elles.

Votre mission

Construire un modèle de scoring qui donnera une prédiction sur la probabilité de faillite d'un client de façon automatique.

Construire un dashboard interactif à destination des gestionnaires de la relation client permettant d'interpréter les prédictions faites par le modèle, et d’améliorer la connaissance client des chargés de relation client.

Mettre en production le modèle de scoring de prédiction à l’aide d’une API, ainsi que le dashboard interactif qui appelle l’API pour les prédictions.

Michaël, votre manager, vous incite à sélectionner un ou des kernels Kaggle pour vous faciliter l’analyse exploratoire, la préparation des données et le feature engineering nécessaires à l’élaboration du modèle de scoring. Si vous le faites, vous devez analyser ce ou ces kernels et le ou les adapterpour vous assurer qu’ils répond(ent) aux besoins de votre mission.

C’est optionnel, mais nous vous encourageons à le faire afin de vous permettre de vous focaliser sur l’élaboration du modèle, son optimisation et sa compréhension.

Spécifications du dashboard

Michaël vous a fourni des spécifications pour le dashboard interactif. Celui-ci devra contenir au minimum les fonctionnalités suivantes :

Permettre de visualiser le score et l’interprétation de ce score pour chaque client de façon intelligible pour une personne non experte en data science.

Permettre de visualiser des informations descriptives relatives à un client (via un système de filtre).

Permettre de comparer les informations descriptives relatives à un client à l’ensemble des clients ou à un groupe de clients similaires.

Spécifications techniques

Michaël vous propose d’utiliser Dash ou Bokeh ou Streamlit pour réaliser le Dahboard interactif.

Michaël souhaite également, afin de pouvoir faire évoluer régulièrement le modèle, tester la mise en oeuvre d’une démarche de type MLOps d’automatisation et d’industrialisation de la gestion du cycle de vie du modèle. Il vous envoie la liste d’outils à utiliser pour créer une plateforme MLOps qui s’appuie sur des outils Open Source. 

Michaël vous demande également de tester l’utilisation de la librairie evidently pour détecter dans le futur du Data Drift en production. Pour cela vous prendrez comme hypothèse que le dataset “application_train” représente les datas pour la modélisation et le dataset “application_test” représente les datas de nouveaux clients une fois le modèle en production. 

L’analyse à l’aide d’evidently vous permettra de détecter éventuellement du Data Drift sur les principales features, entre les datas d’entraînement et les datas de production, au travers du tableau HTML d’analyse que vous aurez réalisé.

Le déploiement de l'application dashboard et de l’API seront réalisées sur une plateforme Cloud, de préférence une solution gratuite, par exemple Azure webapp (ASP F1 gratuit), PythonAnywhere, Heroku avec le package “student” de Github ou tout autre solution.

D’autre part Michaël attend que vous mettiez en oeuvre une démarche d’élaboration des modèles avec Cross-Validation, via GridsearchCV ou équivalent.

Il vous donne un dernier conseil : si vous obtenez des scores supérieurs au 1er du challenge Kaggle (AUC > 0.82), posez-vous la question si vous n’avez pas de l’overfitting dans votre modèle.

Michaël attend également de votre part une note technique, présentant toute votre démarche d’élaboration du modèle jusqu’à l’analyse du Data Drift, afin de partager vos réalisations avec vos collègues. 

Spécifications contextuelles 

Michaël vous fait part de sa vigilance dans l’élaboration du modèle, concernant deux points spécifiques au contexte métier : 

Le déséquilibre entre le nombre de bons et de moins bons clients doit être pris en compte pour élaborer un modèle pertinent, à l’aide d’au moins une méthode au choix

Le déséquilibre du coût métier entre un faux négatif (FN - mauvais client prédit bon client : donc crédit accordé et perte en capital) et un faux positif (FP - bon client prédit mauvais : donc refus crédit et manque à gagner en marge)

Vous pourrez supposer, par exemple, que le coût d’un FN est dix fois supérieur au coût d’un FP

Vous créerez un score “métier” (minimisation du coût d’erreur de prédiction des FN et FP) pour comparer les modèles, afin de choisir le meilleur modèle et ses meilleurs hyperparamètres. Attention cette minimisation du coût métier doit passer par l’optimisation du seuil qui détermine, à partir d’une probabilité, la classe 0 ou 1 (un “predict” suppose un seuil à 0.5 qui n’est pas forcément l’optimum)
En parallèle, maintenez pour comparaison et contrôle des mesures plus techniques, telles que l’AUC et l’accuracy 

### Référentiel d'évaluation

Déployer un modèle via une API dans le Web

CE1 Vous avez défini et préparé un pipeline de déploiement continu.

CE2 Vous avez déployé le modèle de machine learning sous forme d'API (via Flask par exemple) et cette API renvoie bien une prédiction correspondant à une demande.

CE3 Vous avez a mis en œuvre un pipeline de déploiement continu, afin de déployer l'API sur un serveur d'une plateforme Cloud.

CE4 Vous avez a mis en oeuvre des tests unitaires automatisés (par exemple avec pyTest)

CE5 Vous avez a réalisé l'API indépendamment de l'application Dashboard qui utilise le résultat de la prédiction via une requête.

Réaliser un dashboard pour présenter son travail de modélisation

CE1 Vous avez décrit et conçu un parcours utilisateur simple permettant de répondre aux besoins des utilisateurs (les différentes actions et clics sur les différents graphiques permettant de répondre à une question que se pose l'utilisateur).

CE2 Vous avez développé au moins deux graphiques interactifs permettant aux utilisateurs d'explorer les données.

CE3 Vous avez réalisé des graphiques lisibles (taille de texte suffisante, définition lisible).

CE4 Vous avez réalisé des graphiques qui permettent de répondre à la problématique métier.

CE5 Vous avez pris en compte le besoin des personnes en situation de handicap dans la réalisation des graphiques : le candidat doit avoir pris en compte au minimum les critères d'accessibilité du WCAG suivants (https://www.w3.org/Translations/WCAG21-fr/):

Critère de succès 1.1.1 Contenu non textuel

Critère de succès 1.4.1 Utilisation de la couleur

Critère de succès 1.4.3 Contraste (minimum)

Critère de succès 1.4.4 Redimensionnement du texte

Critère de succès 2.4.2 Titre de page

CE6 Vous avez déployé le dashboard sur le web afin qu'il soit accessible pour d'autres utilisateurs sur leurs postes de travail.

 

Présenter son travail de modélisation à l'oral

CE1 Vous avez expliqué de manière compréhensible par un public non technique la méthode d'évaluation de la performance du modèle de machine learning, la façon d'interpréter les résultats du modèle, et la façon d'interpréter l'importance des variables du modèle.

CE2 Vous avez su répondre de manière simple (compréhensible par un public non technique) à au moins une question portant sur sa démarche de modélisation.

CE3 Vous avez présenté une démarche de modélisation et une évaluation complète des modèles, en particulier la comparaison de plusieurs modèles

Rédiger une note méthodologique afin de communiquer sa démarche de modélisation

CE1 Vous avez présenté la démarche de modélisation de manière synthétique dans une note.

CE2 Vous avez explicité la métrique d'évaluation retenue et sa démarche d'optimisation.

CE3 Vous avez explicité l'interprétabilité globale et locale du modèle.

CE4 Vous avez a décrit les limites et les améliorations envisageables pour gagner en performance et en interprétabilité de l'approche de modélisation.

 

Utiliser un logiciel de version de code pour assurer l’intégration du modèle

CE1 Vous avez créé un dossier contenant tous les scripts du projet dans un logiciel de version de code avec Git et l'a partagé avec Github.

CE2 Vous avez présenté un historique des modifications du projet qui affiche au moins trois versions distinctes, auxquelles il est possible d'accéder.

CE3 Vous avez tenu à jour et mis à disposition la liste des packages utilisés ainsi que leur numéro de version.

CE4 Vous avez rédigé un fichier introductif permettant de comprendre l'objectif du projet et le découpage des dossiers.

CE5 Vous avez commenté les scripts et les fonctions facilitant une réutilisation du travail par d'autres personnes et la collaboration.

 
Définir la stratégie d’élaboration d’un modèle d’apprentissage supervisé, et sélectionner et entraîner des modèles adaptés à une problématique métier afin de réaliser une analyse prédictive.

CE1 Vous avez défini sa stratégie d’élaboration d’un modèle pour répondre à un besoin métier

CE2 Vous avez choisi la ou les variables cibles pertinentes.

CE3 Vous avez vérifié qu'il n’y a pas de problème de data leakage (c'est-à-dire, des variables trop corrélées à la variable cible et inconnues a priori dans les données en entrée du modèle).

CE4 Vous avez testé plusieurs algorithmes de façon cohérente, en partant des plus simples vers les plus complexes (au minimum un linéaire et un non linéaire).

 
Évaluer les performances des modèles d’apprentissage supervisé selon différents critères (scores, temps d'entraînement, etc.) en adaptant les paramètres afin de choisir le modèle le plus performant pour la problématique métier.

CE1 Vous avez choisi une métrique adaptée pour évaluer la performance d'un algorithme (par exemple : R2 ou RMSE en régression, accuracy ou AUC en classification, etc.)

Il a mis en oeuvre un score métier pour évaluer les modèles et optimiser les hyperparamètres, qui prend en compte les spécificités du contexte, en particulier le fait que le coût d’un faux négatif et d’un faux positif sont sensiblement différents

CE2 Vous avez exploré d'autres indicateurs de performance que le score pour comprendre les résultats (coefficients des variables en fonction de la pénalisation, visualisation des erreurs en fonction des variables du modèle, temps de calcul...)

CE3 Vous avez séparé les données en train/test pour les évaluer de façon pertinente et détecter l'overfitting

CE4 Vous avez mis en place un modèle simple de référence pour évaluer le pouvoir prédictif du modèle choisi (dummyRegressor ou dummyClassifier)

CE5 Vous avez pris en compte dans sa démarche de modélisation l'éventuel déséquilibre des classes (dans le cas d'une classification).

CE6 Vous avez optimisé les hyper-paramètres pertinents dans les différents algorithmes.

CE7 Vous avez mis en place une validation croisée (via GridsearchCV, RandomizedSearchCV ou équivalent) afin d’optimiser les hyperparamètres et comparer les modèles. Dans le cadre de ce projet : 

Une cross-validation du dataset train est réalisée

Un premier test de différentes valeurs d’hyperparamètres est réalisé sur chaque algorithme testé, et affiné pour l’algorithme final choisi

Tout projet présentant un score AUC anormalement élevé, démontrant de l’overfitting dans le GrisSearchCV, sera invalidé (il ne devrait pas être supérieur au meilleur de la compétition Kaggle : 0.82)

CE8 Vous avez présenté l'ensemble des résultats en allant des modèles les plus simples aux plus complexes. Il a justifié le choix final de l'algorithme et des hyperparamètres.

CE9 Vous avez réalisé l’analyse de l’importance des variables (feature importance) globale sur l’ensemble du jeu de données et locale sur chaque individu du jeu de données.

 
Définir et mettre en œuvre un pipeline d’entraînement des modèles, avec centralisation du stockage des modèles et formalisation des résultats et mesures des différentes expérimentations réalisées, afin d’industrialiser le projet de Machine Learning.

CE1 Vous avez mis en oeuvre un pipeline d’entraînement des modèles reproductible

CE2 Vous avez sérialisé et stocké les modèles créés dans un registre centralisé afin de pouvoir facilement les réutiliser.

CE3 Vous avez formalisé des mesures et résultats de chaque expérimentation, afin de les analyser et de les comparer

 
Définir et mettre en œuvre une stratégie de suivi de la performance d’un modèle en production et en assurer la maintenance afin de garantir dans le temps la production de prédictions performantes.

CE1 Vous avez  défini une stratégie de suivi de la performance du modèle. Dans le cadre du projet : 

choix de réaliser a priori cette analyse sur le dataset disponible : analyse de data drift entre le dataset train et le dataset test

CE2 Vous avez réalisé un système de stockage d’événements relatifs aux prédictions réalisées par l’API et une gestion d’alerte en cas de dégradation significative de la performance. Dans le cadre du projet : 

choix de réaliser a priori cette analyse analyse de data drift, via une simulation dans un notebook et création d’un tableau HTML d’analyse avec la librairie evidently

CE3 Vous avez analysé la stabilité du modèle dans le temps et défini des actions d’amélioration de sa performance. Dans le cadre de ce projet : 

Analyse du tableau HTML evidently, et conclusion sur un éventuel data drift
 
### Compétences évaluées

Définir et mettre en œuvre une stratégie de suivi de la performance d’un modèle

Évaluer les performances des modèles d’apprentissage supervisé

Utiliser un logiciel de version de code pour assurer l’intégration du modèle

Définir la stratégie d’élaboration d’un modèle d’apprentissage supervisé

Réaliser un dashboard pour présenter son travail de modélisation

Rédiger une note méthodologique afin de communiquer sa démarche de modélisation

Présenter son travail de modélisation à l'oral

Déployer un modèle via une API dans le Web

Définir et mettre en œuvre un pipeline d’entraînement des modèles
