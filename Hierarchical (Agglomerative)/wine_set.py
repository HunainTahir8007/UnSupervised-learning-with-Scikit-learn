import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns 
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram,linkage
from umap.umap_ import UMAP



pd.options.display.max_columns=None

df=pd.read_csv('F:\\cssv\\wine.csv')
df.dropna(inplace=True)
df.drop(['Total_Phenols','OD280','Ash'],axis=1,inplace=True)
x=df.values

sca=StandardScaler()
X=sca.fit_transform(x)
#------------plotting the dendenogram -----------------------
link=linkage(X,method='ward')
plt.figure(figsize=(10,5))
dendrogram(link,truncate_mode='level',p=5)
plt.grid()
plt.title("Hierarchical Clustering Dendrogram (Wine Data)")
plt.xlabel("Data points")
plt.ylabel("Distance")
plt.show()
#------------------------Cheaking the value of K --------------------------------
for i in range(2,11):
    mod=AgglomerativeClustering(n_clusters=i,linkage='ward')
    mod.fit(X)
    lab=mod.labels_
    score=silhouette_score(X,lab)
    print("Score for K= ",i," is ",score)


mod=AgglomerativeClustering(n_clusters=4,linkage='ward')
mod.fit(X)
lab=mod.labels_
score=silhouette_score(X,lab)
print(score)

#-------------------------------------------
ua=UMAP(n_components=2,random_state=42)
x_u=ua.fit_transform(X)

plt.figure(figsize=(8,6))
plt.scatter(x_u[:,0], x_u[:,1], c=lab, cmap="plasma", s=80)
plt.title("Hierarchical Clusters (UMAP Reduction)")
plt.grid(linestyle=':', color='grey', linewidth=0.4)
plt.show()