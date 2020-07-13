import pandas as pd
import numpy as np
from functools import reduce
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from statsmodels.stats import weightstats
import seaborn as sns

label_dict = {'B': 0, 
              'M': 1}

def load_data():
    # download data from UCI Machine Learning Repository
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'
    data = pd.read_csv(url, header=None)
    label = data.iloc[:, 1]
    data = data.drop([1], axis=1)
    
    # create the feature names
    names = ['radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness', 'concavity', 'concave points', 'symmetry', 'fractal dimension']
    types = ['mean', 'std', 'worst']
    names = reduce(lambda x,y: x+y, [['%s_%s'%(n1,n2) for n1 in names] for n2 in types])
    
    data.columns = ['id'] + names
    
    return data, label


def clean_data(data):
    # remove id column
    data = data.drop(['id'], axis=1)
    
    # transform to float
    data = data.astype('float')
    
    # scale to [0,1]
    names = data.columns
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    data = pd.DataFrame(data, columns=names)
    
    return data


def divide(x, bin, labels):
    temp = pd.cut(x, bin, labels=labels, right=True)
    temp = [temp.value_counts(i) for i in labels][0]
    res = np.array([temp[i] for i in labels])
    return res

def histogram_score(x, label):
    n = len(x)
    x0 = x[label==0]
    x1 = x[label==1]
    bin = np.arange(0,1,0.1)
    bin[0] = bin[0]-0.01
    labels = np.arange(len(bin)-1)
    s0 = divide(x0, bin, labels) / n
    s1 = divide(x1, bin, labels) / n
    return sum(abs(s0-s1))


def histogram(data, label, name):
    sns.distplot(data[name][label==0], color='darkorange')
    sns.distplot(data[name][label==1], color='lightblue')
    plt.legend(['Benign', 'Mlignant'])


def correlation_heatmap(corr):
    ax = sns.heatmap(
        corr, 
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True,
        annot=True
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )


def hypothesis_testing(data, label, type_of_test):
    if type_of_test == 'z_test':
        f = weightstats.ztest
    else:
        f = weightstats.ttest_ind
    def fun(x):
        l0 = x[label==0]
        l1 = x[label==1]
        return f(l0, l1)[1]
    return data.apply(fun)