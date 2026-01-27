import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Your dataset
data = {
    "age": [25, 27, 29, 31, 45, 47, 49, 51, 60, 62, 64, 66],
    "income": [30, 32, 35, 37, 70, 72, 75, 78, 90, 92, 95, 97],
    "spending_score": [60, 62, 65, 67, 20, 22, 25, 27, 40, 42, 45, 47],
    "education_years": [12, 13, 12, 14, 16, 17, 16, 18, 20, 21, 20, 22],
    "family_size": [3, 2, 4, 3, 5, 6, 4, 5, 2, 3, 4, 2],
    "credit_score": [650, 655, 660, 670, 700, 710, 715, 720, 750, 755, 760, 770],
}
df = pd.DataFrame(data)
X = StandardScaler().fit_transform(df.values)

best_score = -1
best_params = None

#
for ep in np.arange(0.1, 3.0, 0.2):
    for sm in [2, 3, 4, 5]:
        model = DBSCAN(eps=ep, min_samples=sm)
        labels = model.fit_predict(X)

        mask = labels != -1
        if len(set(labels[mask])) > 1:  
            score = silhouette_score(X[mask], labels[mask])
            if score > best_score:
                best_score = score
                best_params = (ep, sm)

print("Best Score:", best_score)
print("Best Parameters (eps, min_samples):", best_params)
eps, min_samples = best_params
model = DBSCAN(eps=eps, min_samples=min_samples)
labels = model.fit_predict(X)

plt.figure(figsize=(6,5))
plt.scatter(df["age"], df["income"], c=labels, cmap="plasma", s=100, edgecolors="k")
plt.xlabel("Age")
plt.ylabel("Income")
plt.title("DBSCAN Clusters (Age vs Income)")
plt.colorbar(label="Cluster ID")
plt.show()