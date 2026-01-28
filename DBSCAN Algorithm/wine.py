import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns 
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from kneed import KneeLocator
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

pd.options.display.max_columns=None

df=pd.read_csv('d:\\cssv\\wine.csv')
df.dropna(inplace=True)
df.drop(['Total_Phenols','OD280','Ash'],axis=1,inplace=True)
x=df.values

sca=StandardScaler()
X=sca.fit_transform(x)

#-------------------cheaking the best hyperparameters----------------
# best_score=-1
# best_params=None
# for eps in np.arange(0.1,5,0.2):
#     for min in [2,3,4,5,6,7,8,9,10,12,14,16,18]:
#         mod=DBSCAN(eps=eps,min_samples=min)
#         mod.fit(X)
#         labels=mod.labels_
#         mask=labels!=-1
#         if len(set(labels[mask]))>1:
#             score=silhouette_score(X[mask],labels[mask])
#             if score>best_score:
#                 best_score=score
#                 best_params=(eps,min)


# print(f"best score :",best_score)
# print("best parameters",best_params)
#---------------------Putting best parameters after chraking-------------------
model=DBSCAN(eps=0.90,min_samples=2)
model.fit(X)
label=model.labels_
print("Score with Noise",silhouette_score(X,label))
mask=label!=-1
if len(set(label[mask]))>1:
    print("Score with out Noise ",silhouette_score(X[mask],label[mask]))

#Model is making the 77% accurate clusters

#--------------------reduction of data for visualization-------------
pc=PCA(n_components=2)
X_pca=pc.fit_transform(X)
plt.scatter(X_pca[label!=-1,0],X_pca[label!=-1,1],c=label[label!=-1],label='without noise', cmap="plasma", s=80)

plt.legend(loc='best')
plt.grid(linestyle=':',color='grey',linewidth='0.4')
plt.title("Plotting the clusters (PCA Reduction)")
plt.show()