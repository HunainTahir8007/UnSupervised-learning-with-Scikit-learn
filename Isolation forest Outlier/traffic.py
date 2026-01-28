import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns 
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score,f1_score,make_scorer
from sklearn.preprocessing import LabelEncoder,StandardScaler
pd.options.display.max_columns=None

df=pd.read_csv('d:\\cssv\\traffic.csv')
df.drop('mean_packet_size',axis=1,inplace=True)


y = df['label']
y = y.replace({0: 1, 1: -1})
x=df

sca=['packet_size', 'inter_arrival_time', 'src_port', 'dst_port',
       'packet_count_5s', 'spectral_entropy', 'frequency_band_energy']
scalar=StandardScaler()
for i in sca:
    x[i]=scalar.fit_transform(x[[i]])

encoder=LabelEncoder()
en=['protocol_type_UDP', 'src_ip_192.168.1.2',
       'src_ip_192.168.1.3', 'dst_ip_192.168.1.5', 'dst_ip_192.168.1.6',
       'tcp_flags_FIN', 'tcp_flags_SYN', 'tcp_flags_SYN-ACK']
for i in en:
    x[i]=encoder.fit_transform(x[i])

X=x.values
#-----------------------------cheaking the best parameters------------------

best_score=-1 
best_params=None
for eps in np.arange(0.2,3,0.2):
    for min in [2,3,4,5,6,7,9,11,13,15]:
        mod=DBSCAN(eps=eps,min_samples=min)
        mod.fit(X)
        labels=mod.labels_
        # maek for removing the noise 
        mask=labels!=-1
        if len(set(labels[mask]))>1:
            score=silhouette_score(X[mask],labels[mask])
            if score>best_score:
                best_score=score
                best_params=(eps,min)
print('Best parameters ',best_params)
print("best score",best_score)
# #-----------------Putting the best values--------------------------

mod=DBSCAN(eps=1,min_samples=2)
mod.fit(X)
labels=mod.labels_
# maek for removing the noise 
mask=labels!=-1
if len(set(labels[mask]))>1:
    score=silhouette_score(X[mask],labels[mask])

print("The best score with out noise is : ",score)

#----------------cheaking the best parameters for the outlier detection -------------

# Here we need a scorer, f1 is common for anomaly detection
scorer = make_scorer(f1_score, pos_label=-1)
parms={
    'n_estimators': [100, 200, 300],
    'max_samples': [0.5, 0.7, 1.0],
    'contamination': [0.01, 0.05, 0.1],
    'max_features': [0.5, 0.7, 1.0]
}
grid=GridSearchCV(
    IsolationForest(),param_grid=parms,  scoring=scorer,
    cv=3,
    n_jobs=-1
)
grid.fit(X,y)
print("Best params:", grid.best_params_)
print("Best score:", grid.best_score_)
#----------------------------------------------------

#----------------putting best params to outlier----------
ios=IsolationForest(contamination=0.05,max_features=0.7,max_samples=1,n_estimators=300)
ios.fit(X)
res = ios.predict(X) 

print("Outliers detected:", np.sum(res == -1))
print("Inliers detected:", np.sum(res == 1))
#--------------------plotting -------------------------------
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[res==1,0], X_pca[res==1,1], c="blue", label="Inliers", alpha=0.6)
plt.scatter(X_pca[res==-1,0], X_pca[res==-1,1], c="red", label="Outliers", alpha=0.6)
plt.legend()
plt.title("Isolation Forest Outlier Detection (PCA Projection)")
plt.show()