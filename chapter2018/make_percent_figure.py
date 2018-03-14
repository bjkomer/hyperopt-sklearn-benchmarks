# Remaking old plots with updated visuals

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

sns.set_style('white')

colors = sns.color_palette() #sns.color_palette("hls", 6)

# Pie chart data
# order is svc, sgd, knn, multinomial_nb, extra_trees, random_forest
classifiers = ['svc', 'sgd', 'knn', 'multinomial_nb', 'extra_trees', 'random_forest']
datasets = ['newsgroups', 'mnist', 'convex']
data = {
    'newsgroups':[33.33, 65, 0, 1.67, 0, 0],
    'mnist':[67.57, 0, 5.41, 0, 13.51, 13.51],
    'convex':[100, 0, 0, 0, 0, 0]
}

data = {
    'svc':[33.33, 67.57, 100],
    'sgd':[65, 0, 0],
    'knn':[0, 5.41, 0],
    'multinomial_nb':[1.67, 0, 0],
    'extra_trees':[0, 13.51, 0],
    'random_forest':[0, 13.51, 0],
}

df = pd.DataFrame(data)

# From raw value to percentage
#totals = [i+j+k for i,j,k in zip(df['newsgroups'], df['mnist'], df['blueBars'])]
totals = [a+b+c+d+e+f for a,b,c,d,e,f in zip(df['svc'], df['sgd'], df['knn'], df['multinomial_nb'], df['extra_trees'], df['random_forest'])]

bars = []
for c in classifiers:
    bars.append([i / j * 100 for i,j in zip(df[c], totals)])



# Plot
bar_width = .8

r = [0, 1, 2]

fig, ax = plt.subplots() 

plt.bar(r, bars[0], color=colors[0], edgecolor='white', width=bar_width)
plt.bar(r, bars[1], color=colors[1], bottom=bars[0], edgecolor='white', width=bar_width)
plt.bar(r, bars[2], color=colors[2], bottom=[i+j for i,j in zip(bars[0], bars[1])], edgecolor='white', width=bar_width)
plt.bar(r, bars[3], color=colors[3], bottom=[i+j+k for i,j,k in zip(bars[0], bars[1], bars[2])], edgecolor='white', width=bar_width)
plt.bar(r, bars[4], color=colors[4], bottom=[i+j+k+l for i,j,k,l in zip(bars[0], bars[1], bars[2], bars[3])], edgecolor='white', width=bar_width)
plt.bar(r, bars[5], color=colors[5], bottom=[i+j+k+l+m for i,j,k,l,m in zip(bars[0], bars[1], bars[2], bars[3], bars[4])], edgecolor='white', width=bar_width)

#plt.xticks(r, datasets)
plt.xticks([.4, 1.4, 2.4], datasets, fontsize=14)
plt.xlabel('Dataset', fontsize=16)

plt.yticks(fontsize=14)
plt.ylabel('Percentage', fontsize=16)

#plt.legend(classifiers, loc='upper right', bbox_to_anchor=(1.25, 1))
plt.legend(classifiers, loc=(1, .5), fontsize=14)

plt.title("Model Selection Distribution", fontsize=20)

sns.despine()
plt.show()
