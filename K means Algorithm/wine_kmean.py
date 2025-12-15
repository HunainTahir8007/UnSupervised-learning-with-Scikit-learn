import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns 
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

pd.options.display.max_columns=None

df=pd.read_csv('F:\\cssv\\wine.csv')
df.dropna(inplace=True)
df.drop(['Total_Phenols','OD280','Ash'],axis=1,inplace=True)
x=df.values

sca=StandardScaler()
X=sca.fit_transform(x)

#---------------------cheaking the best features-------------------
# plt.figure(figsize=(10,6))
# sns.heatmap(df.corr(), annot=False, cmap="coolwarm")
# plt.title("Correlation Heatmap")
# plt.show()


# cheaking the best value of n_clusters
# for i in range(2,20):   
#     mod=KMeans(n_clusters=i,random_state=42)
#     mod.fit(X)
#     label=mod.labels_
#     print(f"score of {i}",silhouette_score(X,label))
#---------------------------------------------------
# best value of n cluster is 3

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
    
mod = KMeans(n_clusters=3, random_state=42)
labels = mod.fit_predict(X_pca)
    
score = silhouette_score(X_pca, labels)
print(f"PCA with best components ",score)

# --- PCA for visualization (2D) ---
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Scatter plot
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap="plasma", s=80, alpha=0.8, edgecolors='k')
plt.title("KMeans Clustering (Wine data, reduced to 2D with PCA)")
plt.xlabel("PCA Feature 1")
plt.ylabel("PCA Feature 2")
plt.grid()
plt.show()