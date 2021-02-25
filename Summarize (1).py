# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.linear_model import LinearRegression as LR
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV


# In[4]:


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sample = pd.read_csv("sample.csv",header=None)


# In[5]:


train.head()


# In[5]:


#describeで要約統計量を取得
train.describe()


# In[6]:


#欠損値を確認
train.isnull().sum()


# In[7]:


#数を確認
train["marital"].value_counts()


# In[8]:


#データの分割を行う
age_bining = pd.cut(train["age"],[0,10,20,30,40,50,60,70,80,90,100])


# In[9]:


age_bining


# In[10]:


#カテゴリー分けした値をグラフで見れるようにする
pout = pd.crosstab(age_bining,train["marital"],margins = True)


# In[11]:


pout


# In[12]:


train.info()


# In[13]:


#今回は三パターンにおける割合を出したが２つの時などは割合が出しやすい
pout["rate"] = pout["single"]/pout["All"]


# In[14]:


pout


# In[7]:


#ilocで横の範囲を指定する
trainX = train.iloc[:,0:17]


# In[8]:


trainX.head()


# In[9]:


y = train["y"]


# In[10]:


testX = test.copy()

# In[11]:


#dummies変数化を行い0or1の値に変換
trainX = pd.get_dummies(trainX)
testX = pd.get_dummies(testX)


# In[12]:


#深さを変えて良いスコアが出るように調整する
clf1 = DT(max_depth =50,min_samples_leaf =1000)


# In[14]:


cross_validate(clf1,trainX,y,cv=5,scoring="roc_auc",n_jobs=-1)


# In[15]:


clf1 = DT(max_depth =50,min_samples_leaf =500)


# In[16]:


cross_validate(clf1,trainX,y,cv=5,scoring="roc_auc",n_jobs=-1)


# In[17]:


clf2 = DT()


# In[18]:


#適した深さを見つけるための操作を行う
parameters = {"max_depth":list(range(2,12))}


# In[19]:


parameters


# In[20]:


gcv = GridSearchCV(clf2,parameters,cv=5,return_train_score=True,scoring="roc_auc",n_jobs=-1)
gcv.fit(trainX,y)


# In[21]:


gcv.cv_results_


# In[22]:


train_score = gcv.cv_results_["mean_train_score"]
test_score = gcv.cv_results_["mean_test_score"]


# In[23]:


train_score


# In[24]:


plt.plot(train_score)
plt.plot(test_score)


# In[25]:


#一番値のいい部分を抽出
gcv.best_params_


# In[26]:


pred = gcv.predict_proba(testX)


# In[27]:


pred = pred[:,1]


# In[28]:


sample[1] = pred


# 

# In[29]:


sample.to_csv("submit.csv",index = None ,header = None)







