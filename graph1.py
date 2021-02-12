#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


train =pd.read_csv("train.csv")
train.head()


# In[5]:


train.isnull()


# In[7]:


train.isnull().any()


# In[8]:


train.isnull().sum()


# In[9]:


train.head()


# In[12]:


train.fillna(0)


# In[14]:


train.dropna(subset=["kcal"])


# In[15]:


train["precipitation"].value_counts()


# In[17]:


train[["y","temperature"]].corr()


# In[18]:


train[["y","kcal"]].corr()


# In[19]:


train.plot.scatter(x="temperature",y="y",figsize=(5,5))


# In[22]:


train.plot.scatter(x="kcal",y="y",figsize=(5,5))


# In[ ]:




