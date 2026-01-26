import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from mlxtend.preprocessing.transactionencoder import TransactionEncoder
import collections as c
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

pd.options.display.max_columns=None
df=pd.read_csv("F:\\cssv\\groceries.csv")
#-----------------removing the nan values--------
 # for this we use the list in list 
 #1st list is for the costumer
 #2nd for the market list

market=[]
for i in range(0,df.shape[0]):
    cust=[]
    for j in df.columns:
        if type(df[j][i])==str:
            cust.append(df[j][i])

    market.append(cust)

#--------------------cheaking the best selling product------------
l=[]
for i in market:
    for j in i:
        l.append(j)

count=c.Counter(l)

dt={"Item NAME":count.keys(),'Count':count.values()}

dic=pd.DataFrame(dt)
dic.sort_values(by=['Count'],ascending=False)
# print(dic)
#---------------------------------------------------
# Preprocessing portion:
pro=TransactionEncoder()
x=pro.fit_transform(market)

df=pd.DataFrame(x,columns=pro.columns_)

mod=apriori(df,min_support=0.05,max_len=3,use_colnames=True).sort_values(by=['support'],ascending=False)
print(mod)

mod["length"] = mod["itemsets"].apply(lambda x: len(x))
print(mod.head(20))

rules = association_rules(mod, metric="lift", min_threshold=1.0)
rules_sorted = rules.sort_values(by="confidence", ascending=False)
print(rules.head(10))
rules_sorted["rule"] = rules_sorted["antecedents"].apply(lambda x: ','.join(list(x))) + " → " + \
                       rules_sorted["consequents"].apply(lambda x: ','.join(list(x)))

# Plot
plt.figure(figsize=(10,6))
plt.barh(rules_sorted["rule"], rules_sorted["confidence"], color="skyblue")
plt.xlabel("Confidence")
plt.ylabel("Association Rule")
plt.title("Top 10 Association Rules by Confidence")
plt.gca().invert_yaxis()  
plt.show()