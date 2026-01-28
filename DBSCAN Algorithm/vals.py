import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

pd.options.display.max_columns=None
df=pd.read_csv("d:\\cssv\\dbscan_data.csv")
x=df.values

sca=StandardScaler()
X=sca.fit_transform(x)

#----------------------cheaking the best hyperparameters--------------
best_score=-1
best_params=None

for ep in np.arange(0.1,4,0.2):
    for min in [2,4,5,6,7,8,9,10,13,15]:
        mod=DBSCAN(eps=ep,min_samples=min)
        mod.fit(X)
        label=mod.labels_
        mark=label!=-1
        if len(set(label[mark]))>1:
            score=silhouette_score(X[mark],label[mark])
            if score>best_score:
                best_score=score
                best_params=(ep,min)
print("best score ",best_score)
print("best parameters",best_params)

#--------------------------------------------------------

#---------------------putting the best values-------------
model=DBSCAN(eps=0.1,min_samples=15)
model.fit(X)
label=model.labels_
mark=label!=-1
if len(set(label[mark]))>1:
    score=silhouette_score(X[mark],label[mark])
print("Best score",score)
#------------------------cheaking feature importance---------------
# sns.heatmap(df.corr())
# plt.show()



#reduction for plotting:-
pc=PCA(n_components=2)
pc=pc.fit_transform(X)

plt.scatter(pc[label!=-1,0],pc[label!=-1,1],c=label[label!=-1],label='3d clusters',cmap='plasma',s=80)
clusters=set(label)-{-1}

for cluster in clusters:
    cluster_points=X[label==cluster]
    centroid=cluster_points.mean(axis=0)
    
    plt.scatter(centroid[0],centroid[1],c='black', s=200, marker='X', edgecolors='white', label='Centroid ')
plt.legend(loc='best')
plt.grid(linestyle=':',color='grey',linewidth='0.4')
plt.title("Plotting the clusters (PCA Reduction)")
plt.show()