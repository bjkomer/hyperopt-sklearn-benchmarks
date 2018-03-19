# Remaking old plots with updated visuals

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

sns.set_style('white')

algo = ['TPE', 'Random'][0]
use_min = True
add_trendline = True
zoom = True

if add_trendline:
    params = {
        'legend.fontsize': 16,
        'axes.labelsize': 16,
        'lines.linewidth': 2,
    }
else:
    params = {
        'legend.fontsize': 16,
        'axes.labelsize': 16,
        'lines.linewidth': 3,
    }
plt.rcParams.update(params)

if algo == 'TPE':
    if use_min:
        fname = 'newsgroups_tpe_min_validation_error.csv'
    else:
        fname = 'newsgroups_tpe_validation_error.csv'
else:
    if use_min:
        fname = 'newsgroups_rand_min_validation_error.csv'
    else:
        fname = 'newsgroups_rand_validation_error.csv'

pd_data = pd.read_csv(
    fname,
    header=None,
    #names=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'],
    #index_col=np.arange(300)+1,
)
print(pd_data)

np_data = np.loadtxt(
    #'newsgroups_tpe_min_validation_error.csv',
    #'newsgroups_rand_min_validation_error.csv',
    fname,
    delimiter=','
)

#print(np_data.shape)

#print(np_data)

#print((np.arange(300)+1).shape)

#data = open('avg_min_newsgroups_tpe.txt', 'r').readlines()

#plt.plot(data)

if add_trendline:
    # Average the data rather than showing the variance
    np_data = np.mean(np_data, axis=1)

sns.tsplot(
    #data=pd_data,
    data=np_data.T,
    time=np.arange(300)+1,
    #value="Validation Error",
)

if add_trendline:
    x = np.arange(300)+1
    z = np.polyfit(x, np_data.T, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), 'r--')

#plt.title("Minimum Validation Error \n TPE Algorithm - 20 Newsgroups Dataset", fontsize=20)
plt.title("{0} Algorithm - 20 Newsgroups Dataset".format(algo))
if use_min:
    plt.suptitle("Minimum Validation Loss", fontsize=20)
else:
    plt.suptitle("Mean Validation Loss", fontsize=20)

plt.xlabel("Evaluations")
plt.ylabel("Validation Loss")

if use_min:
    if zoom:
        # zoomed version
        plt.ylim(0.06, 0.1)
        plt.xlim(0, 300)
    else:
        # non-zoomed version
        plt.ylim(0.05, 0.3)
        plt.xlim(0, 300)
else:
    plt.ylim(0.05, 0.9)
    plt.xlim(0, 300)

sns.despine()
plt.show()
