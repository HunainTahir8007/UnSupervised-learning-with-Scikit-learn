import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np 
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage,dendrogram
from sklearn.preprocessing import StandardScaler
from umap.umap_ import UMAP

data = {
    "Alcohol": [13.2, 12.8, 13.5, 14.1, 12.7, 13.0, 13.6, 14.2],
    "Malic_Acid": [1.8, 2.1, 1.9, 2.5, 1.6, 1.7, 2.0, 2.4],
    "Flavanoids": [2.6, 2.3, 2.8, 3.1, 2.2, 2.5, 2.9, 3.0],
    "Proline": [1050, 980, 1120, 1180, 970, 1020, 1150, 1200]
}

df = pd.DataFrame(data)
X=df.values

sca=StandardScaler()
X=sca.fit_transform(X)

#-----------------plotting the dendenogram--------------
# link=linkage(X,method='ward')
# plt.figure(figsize=(10,6))
# dendrogram(link,truncate_mode='level',p=5)
# plt.grid()
# plt.title("Hierarchical Clustering Dendrogram ")
# plt.xlabel("Data points")
# plt.ylabel("Distance")
# plt.show()
#----------------------------------------
#----------cheaking the best value if the n_clusters----
# best_score=-1
# best_params=[]
# for i in range(2, len(X)):
#     for li in ['average','complete','single','ward']:
#         mod=AgglomerativeClustering(n_clusters=i,linkage=li)
#         mod.fit(X)
#         lab=mod.labels_
#         if len(set(lab))>1:
#          score=silhouette_score(X,lab)
#          if score>best_score:
#             best_score=score
#             best_params=(i,li)
        
# print("Best score",best_score)
# print("best barameters ",best_params)
#----------------------putting the best values------------
mod=AgglomerativeClustering(n_clusters=3,linkage='average')
mod.fit(X)
lab=mod.labels_
score=silhouette_score(X,lab)
print("best score ",score)
#--------------plotting ------------
pp=UMAP(n_neighbors=5,min_dist=0.3,random_state=42)
label=pp.fit_transform(X)
plt.figure(figsize=(8,6))
sns.scatterplot(x=label[:,0], y=label[:,1], hue=lab, palette="Set2", s=100)
plt.title(f"Agglomerative Clustering with UMAP Projection\nSilhouette Score = {score:.3f}")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.show()