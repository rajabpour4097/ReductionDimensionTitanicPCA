import numpy as np
import pandas as pd
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition, datasets

np.random.seed(10)
centers = [
    [1, 1],
    [-1, -1],
    [1, -1]
]
data = pd.read_csv('titanic.csv')

data['Age'].fillna(data['Age'].median(), inplace=True)
numeric_data = data[['Age', 'Fare', 'Pclass', 'Survived', 'SibSp','Parch']]

pre_data=np.array(data['Sex'])
y=pre_data
for i in range(0,891):
    if pre_data[i]=='male':
        y[i]=0
    elif pre_data[i]=='female':
        y[i]=1

X=numeric_data
figure = pyplot.figure(1, figsize=(6, 6))
pyplot.clf()
axes = Axes3D(figure, rect=[0, 0, 0.95, 1], elev=48, azim=134)
pyplot.cla()
pca = decomposition.PCA(n_components=3)
pca.fit(X)
X = pca.transform(X)

for name, label in [('Male', 0), ('Female', 1)]:
    axes.text3D(X[y == label, 0].mean(),
               X[y == label, 1].mean() + 1.5,
               X[y == label, 2].mean(),
               name,
               horizontalalignment='center',
               bbox=dict(alpha=0.5, edgecolor='w', facecolor='w'))

axes.scatter(X[:, 0], X[:, 1], c=y, cmap=pyplot.cm.nipy_spectral, edgecolor='b')
axes.w_xaxis.set_ticklabels([])
axes.w_yaxis.set_ticklabels([])
axes.w_zaxis.set_ticklabels([])
pyplot.show()