# import library

import os
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import dump, load
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix as cfm, precision_score as ps, recall_score as rs, f1_score as f1s, \
    classification_report as cr, accuracy_score as asc
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, Normalizer
from sklearn.utils import resample

from Utils import getData, getBestModelParamsForLR

columnResultName = "Resultado"
#logisticR = LogisticRegression(tol=1e-4, C=10000, max_iter=1000)
#logisticR = LogisticRegression(tol=1e-8, C=8000, max_iter=1000)
# logisticR = LogisticRegression(C=10000, class_weight=None, dual=False, fit_intercept=True,
#                    intercept_scaling=1, l1_ratio=None, max_iter=100,
#                    multi_class='warn', n_jobs=None, penalty='l2',
#                    random_state=None, solver='warn', tol=0.0001, verbose=0,
#                    warm_start=False)
logisticR = LogisticRegression(C=100000, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='warn', n_jobs=None, penalty='l2',
                   random_state=None, solver='warn', tol=0.000001, verbose=0,
                   warm_start=False)


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

    class_names = [0, 1]  # name  of classes
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    # create heatmap
    sns.heatmap(pd.DataFrame(cfm(y_test, prFit)), annot=True, cmap="YlGnBu", fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()

    y_pred_proba = logisticLoaded.predict_proba(x_test)[::, 1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
    plt.legend(loc=4)
    plt.show()

    pr1 = logisticLoaded.predict(fts2)
    print("predico unica", pr1)
    return pr1

def predictLR():
    # buscar arquivo para treinar o modelo
    data = getData("/tmp/predict.csv", "Resultado", False)

    # print(data.head())
    #
    # print(data.isnull().sum())

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

    if len(x_train) < 300:
        #resample somente nos itens de treino!!!
        x_train, y_train = resample(x_train, y_train, n_samples=1000, replace=True)

    print("x_train size:", len(x_train))
    print("x_test size:", len(x_test))

    return predictResult(x_train, y_train, y_test, x_test)[0]
    #return getBestModelParams(x_train, y_train)

def predictWithSavedModel():
    data2 = pd.read_csv("/tmp/predict_result.csv", header=0)
    # vamos percorrer o arquivo com o valor a ser testado, onde vamos pegar as colunas e jogar os valores numa array
    cols2 = data2.columns[(data2.columns != columnResultName)]
    fts2 = data2[cols2]
    fts2 = Normalizer().fit_transform(fts2)

    modelLoaded = load('logistic.model')

    predict = modelLoaded.predict_proba(fts2)
    print("Previsão(porcentagem): ")
    print(predict)

    print("Previsão absoluta: ")
    print(modelLoaded.predict(fts2))

    return predict[0]