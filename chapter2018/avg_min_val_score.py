# Remaking old plots with updated visuals

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

sns.set_style('white')

algo = ['TPE', 'Random'][0]
zoom = True

params = {
    'legend.fontsize': 16,
    'axes.labelsize': 16,
    'lines.linewidth': 3,
}
plt.rcParams.update(params)

if algo == 'TPE':
    pd_data = pd.read_csv(
        'newsgroups_tpe_min_validation_error.csv',
        header=None,
        names=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'],
        #index_col=np.arange(300)+1,
    )
else:
    pd_data = pd.read_csv(
        'newsgroups_rand_min_validation_error.csv',
        header=None,
        names=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'],
        #index_col=np.arange(300)+1,
    )
print(pd_data)

np_data = np.loadtxt(
    #'newsgroups_tpe_min_validation_error.csv',
    'newsgroups_rand_min_validation_error.csv',
    delimiter=','
)

#print(np_data.shape)

#print(np_data)

#print((np.arange(300)+1).shape)

#data = open('avg_min_newsgroups_tpe.txt', 'r').readlines()

#plt.plot(data)

sns.tsplot(
    #data=pd_data,
    data=np_data.T,
    time=np.arange(300)+1,
    #value="Validation Error",
)

#plt.title("Minimum Validation Error \n TPE Algorithm - 20 Newsgroups Dataset", fontsize=20)
plt.title("{0} Algorithm - 20 Newsgroups Dataset".format(algo))
plt.suptitle("Minimum Validation Error", fontsize=20)

plt.xlabel("Evaluations")
plt.ylabel("Validation Error")

if zoom:
    # zoomed version
    plt.ylim(0.06, 0.1)
    plt.xlim(0, 300)
else:
    # non-zoomed version
    plt.ylim(0.05, 0.3)
    plt.xlim(0, 300)

sns.despine()
plt.show()
