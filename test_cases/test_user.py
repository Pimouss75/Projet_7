import joblib
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

model = joblib.load("model.pkl")
data = pd.read_csv("df_train_test.csv")

def test_prediction():

    X_test = data.drop(['TARGET', 'SK_ID_CURR'], axis=1)
    y_test = data.TARGET


    accuracy = model.score(X_test, y_test)

    assert accuracy >= 0.67  # validation du test a partir de 67%
