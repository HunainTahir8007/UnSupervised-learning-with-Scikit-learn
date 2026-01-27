import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

data = {
    "balance": [
        200, 500, 1500, 3000, 250, 700, 1200, 3300, 4000, 100, 
        450, 1700, 2700, 3500, 4200, 800, 1600, 2800, 3900, 50,
        600, 1400, 2600, 3400, 4300, 900, 1800, 2900, 3700, 60
    ],
    "purchases": [
        100, 200, 1500, 50, 80, 300, 1700, 40, 2000, 90, 
        250, 1600, 60, 2100, 70, 400, 1750, 65, 2300, 95,
        350, 1800, 75, 2400, 85, 500, 1900, 55, 2200, 105
    ],
    "cash_advance": [
        0, 300, 0, 400, 0, 200, 0, 500, 0, 0,
        150, 0, 250, 0, 350, 0, 450, 0, 600, 0,
        220, 0, 330, 0, 440, 0, 550, 0, 660, 0
    ],
    "credit_limit": [
        1000, 2000, 3000, 4000, 1200, 2200, 3200, 4200, 5000, 800,
        1800, 2800, 3800, 4800, 5200, 1500, 2500, 3500, 4500, 900,
        1900, 2900, 3900, 4900, 5300, 1600, 2600, 3600, 4600, 1000
    ],
    "payments": [
        300, 800, 2000, 500, 400, 900, 2100, 600, 2500, 350,
        700, 2200, 650, 2700, 700, 1000, 2300, 750, 2800, 370,
        1100, 2400, 800, 2900, 900, 1200, 2500, 850, 3000, 400
    ],
    "tenure": [
        12, 11, 10, 9, 12, 11, 10, 8, 7, 12,
        11, 9, 8, 7, 6, 12, 11, 10, 9, 12,
        11, 8, 7, 6, 5, 12, 11, 10, 9, 12
    ]
}

df=pd.DataFrame(data)
x=df.values

sca=StandardScaler()
X=sca.fit_transform(x)
#------------------------------Cheaking the best parameters-------------------------
# best_score=-1
# best_parms=None
# for ep in np.arange(0.5,10,0.2):
#     for min in [2, 3, 4, 5,6,7,8,9]:
#         mod=DBSCAN(eps=ep,min_samples=min)
#         mod.fit(X)
#         label=mod.labels_
#         mask=label !=-1
#         if len(set(label[mask])) > 1:
#             score=silhouette_score(X[mask],label[mask])
#             if score > best_score:
#                 best_score = score
#                 best_parms = (ep, min)
# print("best score",best_score)
# print("best parametrs",best_parms) 
#------------------model after putting best parameters----------------------
model=DBSCAN(eps=0.5,min_samples=2)
lab=model.fit_predict(X)
print("score with the Noise",silhouette_score(X,lab))
mask = lab != -1
if len(set(lab[mask])) > 1:   # need at least 2 clusters
    score = silhouette_score(X[mask], lab[mask])
    print("Score (without noise):", score)
else:
    print("Not enough clusters to compute silhouette score")

#---------------------plotting---------------------
plt.scatter(X[lab!=-1, 0], X[lab!=-1, 1], c=lab[lab!=-1], cmap="plasma", s=80)
plt.scatter(X[lab==-1, 0], X[lab==-1, 1], c='k', marker='x', s=100) 
# Compute and plot centroids
clusters = set(lab) - {-1}
for cluster in clusters:
    cluster_points = X[lab == cluster]
    centroid = cluster_points.mean(axis=0)
    plt.scatter(centroid[0], centroid[1], c='black', s=200, marker='X', edgecolors='white', label=f'Centroid {cluster}')
plt.title("DBSCAN Clustering (simple)")
plt.xlabel("Feature 1 (scaled)")
plt.ylabel("Feature 2 (scaled)")
plt.show()