#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install -U scikit-learn')


# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[10]:


data = pd.read_csv('churn_prediction_simple.csv')
data.head()
data.info()


# In[12]:


#Separating Independent and dependent variables
X = data.drop(columns = ['churn','customer_id'])
Y = data['churn']


# In[14]:


#Scaling the datasset
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)


# In[18]:


#Splitiing the dataset
from sklearn.model_selection import train_test_split as tts
x_train, x_test, y_train, y_test = tts(scaled_X, Y, train_size = 0.80, stratify = Y, random_state = 101)
x_train.shape, x_test.shape, y_train.shape, y_test.shape


# # model Building, bagging logistic regression

# In[20]:


from sklearn.ensemble import BaggingClassifier as BC
classifier = BC()
classifier.fit(x_train,y_train)


# Bagging of logistics regression

# In[23]:


from sklearn.linear_model import LogisticRegression as LR
classifier = BC(base_estimator = LR(),n_estimators = 150, n_jobs = -1, random_state = 42)
classifier.fit(x_train,y_train)
predicted_values = classifier.predict(x_train)


# In[24]:


from sklearn.metrics import classification_report
print(classification_report(y_train,predicted_values))


# In[25]:


predicted_values = classifier.predict(x_test)
print(classification_report(y_test,predicted_values))


# In[28]:


from sklearn.ensemble import RandomForestClassifier as RFC
classifier = RFC()


# In[30]:


classifier.fit(x_train,y_train)


# In[32]:


from sklearn.metrics import f1_score
def calc_score (model,x1,y1,x2,y2):
    model.fit(x1,y1)
    predict = model.predict(x1)
    f1 = f1_score(y1,predict)
    predict = model.predict(x2)
    f2 = f1_score(y2,predict)
    return f1,f2


# In[35]:


def effect(train_score,test_scorex_axis,title):  # using this function to quickly visualize how the different values of the parameters affect the performance of the model
    plt.figure(figsize = (7,4), dpi = 120)
    plt.plot(x_axis,train_score,color='red',label='train_score')
    plt.plot(x_axis, test_score, color='blue',label='test_score')
    plt.title(title)
    plt.legend()
    plt.xlabel("Parameter_value")
    plt.ylabel("f1_score")
    plt.show()


# # hyperparameter tuning

# In[36]:


classifier = RFC()
classifier.fit(x_train,y_train)


# In[38]:


estimators = [i for i in range(1,600,10)]
train = []
test = []
for i in estimators:
    model = RFC(class_weight = 'balanced_subsample',n_estimators = i, n_jobs = -1, max_depth = 7, random_state = 101)
    f1, f2 = calc_score(model, x_train,y_train,x_test,y_test)
    train.append(f1)
    test.append(f2)


# In[ ]:


effect(train,test,range(1,600,10),'n_estimators')


# # Max_Samples

# In[ ]:


maxsamples = [i/1000 for i in range(1,1000)]
train = []
test = []
for i in maxsamples:
    model = RFC(class_weight = 'balanced_subsample',n_estimator = 50, n_jobs = -1, max_depth = 7, random_state = 101, max_samples = i)
    f1,f2  = calc_score(model,x_train, y_train, x_test, y_test)
    train.append(f1)
    test.append(f2)


# In[ ]:


effect(train, test, maxsamples, 'bootstrap sample fraction')


# In[ ]:


maxfeatures = range(1,X.shape[1])
train=[]
test=[]
for i in maxfeatures:
    model = RFC(class_weight = 'balanced_subsample', n_estimators = 50, n_jobs = -1, max_depth = 7, random_state = 101, max_features = i)
    f1,f2 = calc_score = (model,x_train,y_train,x_test,y_test)
    train.append(f1)
    test.append(f2)
    


# In[ ]:


effect (train,test,maxfeatures,'number of max features for individual trees')


# In[ ]:




