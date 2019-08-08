import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


def plotCount(data, resultColumn):
    target_count = data[resultColumn].value_counts()
    print('Class 0:', target_count[0])
    print('Class 1:', target_count[1])
    print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')

    target_count.plot(kind='bar', title='Count (target)')
    plt.show()


def getData(csvName, resultColumn, shouldPlotCount):
    data = pd.read_csv(csvName, header=0)
    data.info()
    if shouldPlotCount:
        plotCount(data, resultColumn)
    return data


def getBestModelParamsForLR(x_train, y_train):
    params = {'C': [1, 10, 100, 1000, 10000],
              'tol': [1e-2, 1e-4, 1e-6, 1e-8, 1e-10],
              'solver': ['lbfgs', 'liblinear']}
    # carrying out grid search
    logisticValidation = LogisticRegression()
    clf = GridSearchCV(logisticValidation, params)
    clf.fit(x_train, y_train)
    # the selected parameters by grid search
    print(clf.best_estimator_)
    return [0]


def getBestModelParamsForSVM(x_train, y_train):
    params = {'C': [1, 10, 100, 1000, 10000]}
    # carrying out grid search
    svm = SVC()
    clf = GridSearchCV(svm, params)
    clf.fit(x_train, y_train)
    # the selected parameters by grid search
    print(clf.best_estimator_)
    return [0]