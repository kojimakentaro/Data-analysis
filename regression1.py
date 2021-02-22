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


# In[3]:


train["y"].plot(figsize=(12,4))


# In[4]:


train.head()


# In[7]:


train["year"]= train["datetime"].apply(lambda x :x.split("-")[0])
train["month"]=train["datetime"].apply(lambda x :x.split("-")[1])


# In[8]:


train.head()


# In[9]:


test["year"]=test["datetime"].apply(lambda x:x.split("-")[0])
test["month"]=test["datetime"].apply(lambda x :x.split("-")[1])


# In[10]:


test.head()


# In[11]:


train.info()


# In[13]:


train["year"] =train["year"].astype(np.int)
train["month"]=train["month"].astype(np.int)
test["year"]=test["year"].astype(np.int)
test["month"]=test["month"].astype(np.int)


# In[14]:


train.info()


# In[15]:


trainX=train[["year","month"]]
testX=test[["year","month"]]


# In[16]:


trainX.head()


# In[17]:


y =train["y"]


# In[18]:


model1 =LR()


# In[19]:


model1.fit(trainX,y)


# In[20]:


model1.coef_


# In[21]:


model1.intercept_


# In[22]:


pred=model1.predict(testX)


# In[23]:


sample1=pred


# In[24]:


sample.to_csv("submit4.csv",index=None,header=None)


# In[25]:


pred=model1.predict(trainX)


# In[26]:


train["pred"]=pred


# In[27]:


train["res"]=train["y"]-train["pred"]


# In[28]:


train.sort_values(by="res")


# In[30]:


def jisaku1(x):
    if x=="お楽しみメニュー":
        return 1
    else:
        return 0


# In[33]:


train["fun"] =train["remarks"].apply(lambda x:jisaku1(x))
test["fun"]=test["remarks"].apply(lambda x:jisaku1(x))


# In[35]:


train


# In[39]:


trainX =train[["year","month","fun","temperature"]]
testX=test[["year","month","fun","temperature"]]


# In[40]:


model2 =LR()


# In[41]:


model2.fit(trainX,y)


# In[42]:


model2.coef_


# In[45]:


model2.intercept_


# In[47]:


pred2 = model2.predict(testX)


# In[48]:


sample[1]=pred2


# In[49]:


sample.to_csv("submit5.csv",index=None,header=None)


# In[ ]:




