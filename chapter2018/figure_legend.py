# Remaking old plots with updated visuals

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

sns.set_style('white')

"""
colors = sns.color_palette() #sns.color_palette("hls", 6)
colors = sns.color_palette("hls", 7)
# rearrange
colors = [
    colors[6],
    colors[4],
    colors[0],
    colors[2],
    colors[5],
    colors[1],
    colors[3],
]
#colors = colors[1:]
"""
colors = sns.color_palette() #sns.color_palette("hls", 6)

# Rearrange so a different colour is used for 'any' and the rest are consistent with the percent plot
#colors = [colors[-1]] + colors[0:-1]

colors2 = sns.color_palette("hls", 7)
# rearrange
colors2 = [
    colors2[6],
    colors2[4],
    colors2[0],
    colors2[2],
    colors2[5],
    colors2[1],
    colors2[3],
]

colors = [colors2[0]] + colors

sns.set_palette(colors)

classifiers = ['Any Classifier', 'Support Vector Classifier', 'Stochastic Gradient Descent', 'K-Nearest Neighbors', 'Multinomial Naive Bayes', 'Extra Trees', 'Random Forest']

bar_width=1
num = len(classifiers)

for i in range(num):
    plt.bar(np.arange(num), np.ones(num), color=colors[i], edgecolor='white', width=bar_width)

# Remove the legend for the combined figure
plt.legend(classifiers, loc=(1, .5), fontsize=24)

# Remove the title for the combined figure
#plt.title("Model Selection Distribution", fontsize=20)

sns.despine()
plt.show()
