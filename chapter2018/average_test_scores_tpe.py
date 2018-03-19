# Remaking old plots with updated visuals

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

sns.set_style('white')
#sns.set(font_scale=2)

params = {
    'legend.fontsize': 16,
    'axes.labelsize': 16,
}
plt.rcParams.update(params)

colors = sns.color_palette() #sns.color_palette("hls", 6)

# Rearrange so a different colour is used for 'any' and the rest are consistent with the percent plot
colors = [colors[-1]] + colors[0:-1]

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

colors[0] = colors2[0]

sns.set_palette(colors)

#classifiers = ['any', 'svc', 'sgd', 'knn', 'multinomial_nb']
classifiers = ['Any Classifier', 'Support Vector Classifier', 'Stochastic Gradient Descent', 'K-Nearest Neighbors', 'Multinomial Naive Bayes']

# Average score results
newsgroups = [0.8435264013, 0.8242842637, 0.8446193367, 0.6549690476, 0.8294537354]
mnist = [0.9787434348, 0.9805307388, 0.9129817647, 0.9775175958, 0.8235408847]
convex = [0.840266182, 0.8578079601, 0.6114531703, 0.6421469594, 0.5021988586]

df = pd.DataFrame()

for i, clf in enumerate(classifiers):
    df = df.append({'Classifier': clf, 'Score': newsgroups[i], 'Dataset': 'newsgroups'}, ignore_index=True)
    df = df.append({'Classifier': clf, 'Score': mnist[i], 'Dataset': 'mnist'}, ignore_index=True)
    df = df.append({'Classifier': clf, 'Score': convex[i], 'Dataset': 'convex'}, ignore_index=True)

#sns.barplot([newsgroups, mnist, convex])
#sns.factorplot([newsgroups, mnist, convex])

#sns.factorplot(data=df, x='Dataset', y='Score', hue='Classifier', kind='bar')
# Remove the legend for the combined figure
sns.factorplot(data=df, x='Dataset', y='Score', hue='Classifier', kind='bar', legend=False)


plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
#plt.legend(fontsize=16)
#plt.ylabel(fontsize=16)
#plt.xlabel(fontsize=16)


# Remove the title for the combined figure
#plt.title("Average Test Scores", fontsize=20)

plt.show()
