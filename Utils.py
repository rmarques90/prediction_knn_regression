import pandas as pd
from matplotlib import pyplot as plt


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
