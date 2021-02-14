#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
sample=pd.read_csv("sample.csv",header=None)


# In[6]:


train.head()


# In[8]:


train.shape


# In[9]:


train.describe()


# In[10]:


test.describe()


# In[11]:


train.isnull().sum()


# In[12]:


test.isnull().sum()


# In[14]:


train["y"].value_counts()


# In[16]:


pd.crosstab(train["marital"],train["y"],margins=True)


# In[17]:


train["age"].describe()


# In[20]:


age_bining=pd.cut(train["age"],[0,20,30,40,50,60,100])


# In[21]:


age_bining


# In[23]:


pd.crosstab(age_bining,train["y"],margins=True)


# In[24]:


test.shape


# In[25]:


train.info()


# In[26]:


pout =pd.crosstab(train["poutcome"],train["y"],margins=True)


# In[27]:


pout


# In[28]:


pout["rate"]=pout[1]/pout["All"]


# In[29]:


pout


# In[30]:


train["duration"].describe()


# In[31]:


duration_bining=pd.cut(train["duration"],[-1,100,200,300,400,500,600,700,800,900,1000])


# In[32]:


duration_bining


# In[35]:


dura=pd.crosstab(duration_bining,train["y"],margins=True)


# In[36]:


dura


# In[37]:


dura["rate"]=dura[1]/dura["All"]


# In[38]:


dura


# In[ ]:




