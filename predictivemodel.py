#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import LinearRegression as LR


# In[2]:


train = pd.read_csv("train.csv")
test =pd.read_csv("test.csv")
sample =pd.read_csv("sample.csv",header=None)


# In[5]:


train.head()


# In[7]:


trainX = train["temperature"]
y = train["y"]


# In[8]:


testX = test["temperature"]


# In[9]:


trainX = trainX.values.reshape(-1,1)
testX =testX.values.reshape(-1,1)


# In[10]:


model1 =LR()


# In[12]:


model1.fit(trainX,y)


# In[14]:


model1.coef_


# In[15]:


model1.intercept_


# In[16]:


pred = model1.predict(testX)


# In[17]:


pred


# In[18]:


sample.head()


# In[19]:


sample[1] =pred


# In[21]:


sample.head()


# In[22]:


sample.to_csv("submit1.csv",index=None,header=None)


# In[25]:


train.isnull


# In[27]:


test.isnull


# In[29]:


avg = train["kcal"].mean()


# In[30]:


avg


# In[31]:


trainX =train["kcal"].fillna(avg)
testX =test["kcal"].fillna(avg)


# In[32]:


testX


# In[33]:


trainX =trainX.values.reshape(-1,1)
testX =testX.values.reshape(-1,1)


# In[34]:


y = train["y"]


# In[35]:


model2 = LR()


# In[36]:


model2.fit(trainX,y)


# In[37]:


model2.coef_


# In[38]:


model2.intercept_


# In[39]:


pred2 = model2.predict(testX)


# In[41]:


sample[1]= pred2


# In[42]:


sample.head()


# In[43]:


sample.to_csv("submit2.csv",index=None,header=None)


# In[ ]:




