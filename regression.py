#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import LinearRegression as LR


# In[3]:


train = pd.read_csv("train.csv")
test =pd.read_csv("test.csv")
sample =pd.read_csv("sample.csv",header=None)


# In[4]:


train["week"].value_counts()


# In[5]:


pd.get_dummies(train["week"])


# In[7]:


trainX=pd.get_dummies(train[["week","temperature"]])


# In[8]:


trainX.head()


# In[9]:


y =train["y"]


# In[10]:


model =LR()


# In[11]:


model.fit(trainX,y)


# In[12]:


model.coef_


# In[13]:


model.intercept_


# In[15]:


testX =pd.get_dummies(test[["week","temperature"]])


# In[16]:


testX.head()


# In[17]:


pred = model.predict(testX)


# In[18]:


pred


# In[19]:


sample[1]=pred


# In[21]:


sample.to_csv("submit3.csv",index=None,header=None)


# In[ ]:




