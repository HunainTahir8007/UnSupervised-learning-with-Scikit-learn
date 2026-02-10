import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd 
import collections as c 
from mlxtend.preprocessing.transactionencoder import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules

pd.options.display.max_columns=None 
df=pd.read_csv("d:\\cssv\\groceries.csv")

#------------------------removing the nan values------------
market=[]
for i in range(0,df.shape[0]):
    cust=[]
    for j in df.columns:
        if type(df[j][i])==str:
            cust.append(df[j][i])

    market.append(cust)

#------------------------------------------------------
l=[]
for i in market:
    for j in i:
       l.append(j)

col=c.Counter(l)

dd=pd.DataFrame({'Items':col.keys(),"values":col.values()}).sort_values(by=['values'],ascending=False)
# print(dd)
#---------------------------Prepropessing-------------------
pre=TransactionEncoder()
x=pre.fit_transform(market)
df=pd.DataFrame(x,columns=pre.columns_)

#Appling the model:
model=fpgrowth(df,min_support=0.05,max_len=3,use_colnames=True).sort_values(by=['support'])
# print(model)
model['len']=model['itemsets'].apply(lambda x: len(x))
print(model.head(20))

#applying the association rules:

rules=association_rules(model,metric='lift',min_threshold=1)
rules_sorted=rules.sort_values(by='confidence',ascending=False)
print(rules_sorted.head(20))

#--------------------------plotting -------------------
rules_sorted['rule'] = (
    rules_sorted["antecedents"].apply(lambda x: ','.join(list(x))) 
    + " → " 
    + rules_sorted["consequents"].apply(lambda x: ','.join(list(x)))
)
plt.barh(rules_sorted["rule"], rules_sorted["confidence"], color="skyblue")
plt.xlabel("Confidence")
plt.ylabel("Association Rule")
plt.title("Top 10 Association Rules by Confidence")
plt.gca().invert_yaxis()  
plt.show()