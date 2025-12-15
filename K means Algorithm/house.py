import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score,silhouette_score
import seaborn as sns
from sklearn.preprocessing import StandardScaler,LabelEncoder

from kneed import KneeLocator
pd.options.display.max_columns=None

df=pd.read_csv('d:\\cssv\\Housing.csv')
x=df.drop("price",axis=1)

print(x.columns)

lab=LabelEncoder()
vals=['mainroad', 'guestroom',
       'basement', 'hotwaterheating', 'airconditioning','prefarea',
       'furnishingstatus']
for i in vals:
    x[i]=lab.fit_transform(x[i])

sca=StandardScaler()
sc=['area', 'bedrooms', 'bathrooms', 'stories','parking']
for i in sc:
    x[i]=sca.fit_transform(x[[i]])

X=x.values

km = KMeans(n_clusters=2, random_state=42)
km.fit(X)
score = silhouette_score(X, km.labels_)
print('Score:',score)


#----------------------------------cheaking the best value for the cluster--------------

# kneedle = KneeLocator(range(1,20), ws, curve="convex", direction="decreasing")
# print("Elbow point (best k):", kneedle.elbow)

# plt.plot(range(1,20), ws, marker='o')
# plt.axvline(kneedle.elbow, color='r', linestyle='--')
# plt.legend()
# plt.show()
#---------------------------------------------------------------------------------------
#  conclusion :
   # n_clusters=2 is the best value for the model our model is not in the perfect clustering shape
   # so the silhouette_score is low
   #while the code and model is correct
   