# import library

import numpy as np
from joblib import dump, load
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix as cfm, precision_score as ps, recall_score as rs, f1_score as f1s, \
    classification_report as cr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, Normalizer

from Utils import getData

columnResultName = "Resultado"
# randomForest = RandomForestClassifier(n_estimators=200, min_samples_leaf=0.4, min_samples_split=5, max_depth=300)
randomForest = RandomForestClassifier(n_estimators=2000, max_depth=100, min_samples_leaf=5, min_samples_split=10)


def predictResult(x_train, y_train, y_test, x_test):
    data2 = pd.read_csv("/tmp/predict_result.csv", header=0)
    # vamos percorrer o arquivo com o valor a ser testado, onde vamos pegar as colunas e jogar os valores numa array
    cols2 = data2.columns[(data2.columns != columnResultName)]
    fts2 = data2[cols2]
    fts2 = Normalizer().fit_transform(fts2)

    randomForest.fit(x_train, y_train)

    dump(randomForest, 'randomForest.model')

    randomForestLoaded = load('randomForest.model')
    prFit = randomForestLoaded.predict(x_test)
    print("predicao:", prFit)
    print("Matriz de Confusao LR:")
    print(cfm(y_test, prFit))
    print("F1 score LR:")
    print(f1s(y_test, prFit))
    print("Precision score LR:")
    print(ps(y_test, prFit))
    print("Recall score LR:")
    print(rs(y_test, prFit))
    print("Classification Report")
    print(cr(y_test, prFit))

    pr1 = randomForestLoaded.predict(fts2)
    print("predico unica", pr1)
    return pr1

def predictRF():
    # buscar arquivo para treinar o modelo
    data = getData("/tmp/predict.csv", "Resultado", False)

    print(data.head())

    print(data.isnull().sum())

    # criando os targets
    y = np.array(data[columnResultName])
    labels = LabelEncoder()
    # transformando os targets em números 0,1,2...
    target = labels.fit_transform(y)

    cols = data.columns[(data.columns != columnResultName)]
    features = data[cols]
    # features = (features - features.mean()) / (features.std())
    # normalizar as features, para evitar uma se destacar diante da outra
    #features = (features - features.min()) / (features.max() - features.min())
    features = Normalizer().fit_transform(features)

    # separar o database em treino/teste
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.33, random_state=42)
    print("x_train size:", len(x_train))
    print("x_test size:", len(x_test))

    return predictResult(x_train, y_train, y_test, x_test)[0]
