#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.tree import export_graphviz
import pydotplus
from IPython.display import Image


# In[4]:


train =pd.read_csv("train.csv")
test =pd.read_csv("test.csv")
sample=pd.read_csv("sample.csv",header=None)


# In[5]:


train.head()


# In[6]:


trainX=train.iloc[:,0:17]


# In[8]:


trainX.head()


# In[9]:


y=train["y"]


# In[10]:


test.head()


# In[11]:


testX=test.copy()


# In[12]:


trainX=pd.get_dummies(trainX)
testX=pd.get_dummies(testX)


# In[13]:


clf1=DT(max_depth=2,min_samples_leaf=500)


# In[14]:


clf1.fit(trainX,y)


# In[18]:


export_graphviz(clf1,out_file="tree.dot",feature_names=trainX.columns,class_names=["0","1"],filled=True,rounded=True)


# In[19]:


g=pydotplus.graph_from_dot_file(path="tree.dot")
Image(g.create_png())


# In[20]:


pred=clf1.predict_proba(testX)


# In[22]:


pred = pred[:,1]


# In[23]:


sample[1]=pred


# In[24]:


sample.to_csv("sabmit_bank.csv",index=None,header=None)


# In[25]:


clf2 = DT(max_depth=4,min_samples_leaf=500)


# In[26]:


clf2.fit(trainX,y)


# In[27]:


export_graphviz(clf2,out_file="tree2.dot",feature_names=trainX.columns,class_names=["0","1"],filled=True,rounded=True)


# In[28]:


g=pydotplus.graph_from_dot_file(path="tree2.dot")
Image(g.create_png())


# In[29]:


pred2 =clf2.predict_proba(testX)


# In[30]:


pred2 =pred2[:,1]


# In[31]:


sample[1]=pred2


# In[33]:


sample.to_csv("submit2.csv",index=None,header=None)


# In[ ]:




