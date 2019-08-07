# import library

import os
import numpy as np
import pandas as pd
from joblib import dump, load
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix as cfm, precision_score as ps, recall_score as rs, f1_score as f1s, \
    classification_report as cr
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

from Utils import getData

columnResultName = "Resultado"
knn = KNeighborsClassifier()

def checkKnnNumbers(x_train, y_train, x_test, shouldPlot):
    return 3
    maxRange = 20
    if len(x_test) < 20:
        maxRange = len(x_test)
    k_range = list(range(1, maxRange))
    k_scores = []
    k_n_x_scores = {}
    for k in k_range:
        knn.n_neighbors = k
        scores = cross_val_score(knn, x_train, y_train, cv=3)
        k_scores.append(scores.mean())
        k_n_x_scores[k] = {
            "score": scores.mean()
        }
    # print(np.round(k_scores, 3))

    print("score map", k_n_x_scores)
    betterNeighborCount = max(k_n_x_scores, key=lambda x: k_n_x_scores[x]["score"])
    print("better N", betterNeighborCount)
    if shouldPlot:
        plt.plot(k_range, k_scores, color='red')
        plt.xlabel('Valores de K')
        plt.ylabel('Recall')
        plt.show()
    return betterNeighborCount


def predictResult(betterN, x_train, y_train, y_test, x_test):
    data2 = pd.read_csv("/tmp/predict_result.csv", header=0)
    # vamos percorrer o arquivo com o valor a ser testado, onde vamos pegar as colunas e jogar os valores numa array
    cols2 = data2.columns[(data2.columns != columnResultName)]
    fts2 = np.array(data2[cols2])

    #quando nao mandar um vaor de betterN, significa que demos o load do modelo
    if betterN > 0:
        knn.n_neighbors = betterN
        knn.fit(x_train, y_train)

        # dump(knn, 'models/knn_teste.joblib')

        prFit = knn.predict(x_test)
        print("predicao: a", prFit)
        print("Matriz de Confusao NB:")
        print(cfm(y_test, prFit))
        print("F1 score NB:")
        print(f1s(y_test, prFit))
        print("Precision score NB:")
        print(ps(y_test, prFit))
        print("Recall score NB:")
        print(rs(y_test, prFit))
        print("Classification Report")
        print(cr(y_test, prFit))

    pr1 = knn.predict(fts2)
    print("predico unica", int(pr1[0]))
    print("predicao unica score")
    print(pr1)
    return pr1

def predict():
    # buscar arquivo para treinar o modelo
    data = getData("/tmp/predict.csv", "Resultado", False)

    # criando os targets
    y = np.array(data[columnResultName])
    labels = LabelEncoder()
    target = labels.fit_transform(y)

    cols = data.columns[(data.columns != columnResultName)]
    features = data[cols]
    features = (features - features.mean()) / (features.std())

    # separar o database em treino/teste
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.33, random_state=42)
    print("x_train size:", len(x_train))
    print("x_test size:", len(x_test))

    betterN = checkKnnNumbers(x_train, y_train, x_test, False)
    return predictResult(betterN, x_train, y_train, y_test, x_test)[0]

    # if not os.path.exists('models/knn_teste.joblib'):
    #     betterN = checkKnnNumbers(x_train, y_train, x_test, False)
    #     return predictResult(betterN, x_train, y_train, y_test, x_test)[0]
    # else:
    #     knn = load('models/knn_teste.joblib')
    #     return predictResult(0, x_train, y_train, y_test, x_test)[0]