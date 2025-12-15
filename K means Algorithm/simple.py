import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans

data = {
    "X": [1, 1, 1, 10, 10],
    "Y": [2, 4, 0, 2, 4]
}

df=pd.DataFrame(data)
X=df.values

model=KMeans(n_clusters=2,random_state=42)
label=model.fit_predict(X)
print(label)