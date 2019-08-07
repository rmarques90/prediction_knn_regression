# import library

import os
import numpy as np
import pandas as pd
from joblib import dump, load
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix as cfm, precision_score as ps, recall_score as rs, f1_score as f1s, \
    classification_report as cr, accuracy_score as asc
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, Normalizer

from Utils import getData

columnResultName = "Resultado"
#logisticR = LogisticRegression(tol=1e-4, C=10000, max_iter=1000)
logisticR = LogisticRegression(tol=1e-4, C=10000, max_iter=1000)


def predictResult(x_train, y_train, y_test, x_test):
    data2 = pd.read_csv("/tmp/predict_result.csv", header=0)
    # vamos percorrer o arquivo com o valor a ser testado, onde vamos pegar as colunas e jogar os valores numa array
    cols2 = data2.columns[(data2.columns != columnResultName)]
    fts2 = data2[cols2]
    fts2 = Normalizer().fit_transform(fts2)

    scores = cross_val_score(logisticR, x_train, y_train, n_jobs=30)
    print("scores cross val")
    print(scores)

    logisticR.fit(x_train, y_train)
    dump(logisticR, 'logistic.model')

    logisticLoaded = load('logistic.model')


    prFit = logisticLoaded.predict(x_test)
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
    print("Accuracy score")
    print(asc(y_test, prFit))

    results = {}
    pr1 = None
    for x in range(5):
        pr1 = logisticLoaded.predict(fts2)
        results[x] = {"score": pr1}

    print("predico unica", results)
    return pr1

def predictLR():
    # buscar arquivo para treinar o modelo
    data = getData("/tmp/predict.csv", "Resultado", False)

    print(data.head())

    print(data.isnull().sum())

    # criando os targets
    y = np.array(data[columnResultName])
    labels = LabelEncoder()
    target = labels.fit_transform(y)

    cols = data.columns[(data.columns != columnResultName)]
    features = data[cols]
    # normalizar as features, para evitar uma se destacar diante da outra
    # features = (features - features.min()) / (features.max() - features.min())
    # features = (features - features.mean()) / (features.std())
    features = Normalizer().fit_transform(features)

    # separar o database em treino/teste
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.33, stratify=target)
    print("x_train size:", len(x_train))
    print("x_test size:", len(x_test))

    return predictResult(x_train, y_train, y_test, x_test)[0]

def predictWithSavedModel():
    data2 = pd.read_csv("/tmp/predict_result.csv", header=0)
    # vamos percorrer o arquivo com o valor a ser testado, onde vamos pegar as colunas e jogar os valores numa array
    cols2 = data2.columns[(data2.columns != columnResultName)]
    fts2 = data2[cols2]
    fts2 = Normalizer().fit_transform(fts2)

    modelLoaded = load('logistic.model')

    predict = modelLoaded.predict(fts2)
    print("Previsão: ")
    print(predict)
    return predict[0]