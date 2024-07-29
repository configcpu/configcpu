from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = load_iris();

x = pd.DataFrame(dataset.data);
x.columns = ['Sepal_Length' , 'Sepal_Width' ,'Petal_Length' , 'Petal_Width'];
y = pd.DataFrame(dataset.target)
y.columns = ['Target'];

plt.figure(figsize=(14,7));
colormap = np.array(['red' , 'lime' , 'black']);

plt.subplot(1,3,1);
plt.scatter(x.Petal_Length , x.Petal_Width , c =colormap[y.Target]);
plt.title('Real');

plt.subplot(1,3,2);
model = KMeans(n_clusters=3);
model.fit(x);
predy = np.choose(model.labels_ , [0,1,2]);
plt.scatter(x.Petal_Length , x.Petal_Width , c=colormap[predy]);
plt.title('KMeans');

scaler = StandardScaler();
scaler.fit(x);
xsa = scaler.transform(x)
xs = pd.DataFrame(xsa , columns = x.columns);
gmm = GaussianMixture(n_components=3);
gmm.fit(xs)
y_cluster_gmm = gmm.predict(xs);
plt.subplot(1,3,3);

plt.scatter(x.Petal_Length , x.Petal_Width , c=colormap[y_cluster_gmm]);
plt.title('GMM classification');
plt.show();

