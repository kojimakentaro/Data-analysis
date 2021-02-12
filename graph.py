#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


train = pd.read_csv("train.csv")


# In[5]:


train.head()


# In[6]:


train["y"].plot()


# In[7]:


train["y"].plot(figsize=(12,4))


# In[13]:


train["y"].plot(figsize=(12,4),title="graph")


# In[15]:


ax = train["y"].plot(figsize=(12,4),title="graph")
ax.set_xlabel("time")
ax.set_ylabel("y")


# In[17]:


train["y"].plot.hist(grid=True)


# In[19]:


plt.axvline(x=5,color="red")


# In[25]:


plt.axvline(x=train["y"].mean(),color="red")
train["y"].plot.hist(figsize=(12,4),title="histogram")    
plt.savefig("sample_fig.png")


# In[26]:


train[["y","week"]].boxplot(by="week")


# In[27]:


ax=train["temperature"].plot(title="temperature")
ax.set_xlabel("time")
ax.set_ylabel("temperature")


# In[28]:


plt.axvline(x=train["kcal"].mean(),color="red")
train["kcal"].plot.hist()


# In[30]:


train[["y","weather"]].boxplot(by="weather")


# In[ ]:




