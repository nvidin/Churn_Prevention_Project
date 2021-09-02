#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# In[2]:


training_data=pd.read_csv('ml_case_training_data.csv')
churn_data=pd.read_csv('ml_case_training_output.csv')
historic_data=pd.read_csv('ml_case_training_hist_data.csv')


# In[138]:


training_data.head(10)
training_data.info()
training_data.describe().T


# In[4]:


training_data['activity_new'].value_counts(dropna=False)


# In[5]:


training_data['channel_sales'].value_counts(dropna=False)


# In[6]:


training_data['channel_sales'].nunique(dropna=False)


# In[3]:


training_data['has_gas'].value_counts(dropna=False)


# In[4]:


training_data['has_gas']=training_data['has_gas'].replace(('f','t'),(0,1))


# In[5]:


train=pd.merge(training_data,churn_data,on='id')


# In[61]:


train.isnull().sum()/len(train.index)*100


# In[10]:


null_values=train.isnull().sum()/len(train.index)*100


# In[11]:


null_values.plot(kind='bar')


# In[12]:


churn=train[['id','churn']]


# In[13]:


churn


# In[14]:


total_churn=churn.groupby('churn').count()


# In[15]:


total_churn


# In[16]:


churn_percentage=total_churn/total_churn.sum()*100


# In[17]:


churn_percentage


# In[25]:


churn_percentage.T.plot(kind='bar',stacked=True,xlabel='Companies')


# In[6]:


train.drop(columns=["campaign_disc_ele", "date_first_activ", "forecast_base_bill_ele","forecast_base_bill_year",
"forecast_bill_12m", "forecast_cons","activity_new","channel_sales","origin_up"], inplace=True)


# In[7]:


train.info()


# In[29]:


train[train.duplicated()]


# In[8]:


train['date_activ']=pd.to_datetime(train['date_activ'])
train['date_end']=pd.to_datetime(train['date_end'])
train['date_modif_prod']=pd.to_datetime(train['date_modif_prod'])
train['date_renewal']=pd.to_datetime(train['date_renewal'])


# In[9]:


historic_data.info()
train['date_end']=train['date_end'].fillna(train['date_end'].value_counts().index[0])
train['date_modif_prod']=train['date_modif_prod'].fillna(train['date_modif_prod'].value_counts().index[0])
train['date_renewal']=train['date_renewal'].fillna(train['date_renewal'].value_counts().index[0])


# In[10]:


historic_data['price_date']=pd.to_datetime(historic_data['price_date'])


# In[11]:


columns=historic_data.select_dtypes('float64')


# In[12]:


for x in columns:
    historic_data[x]=historic_data[x].fillna(historic_data[x].median())


# In[13]:


historic_data.describe().T


# In[14]:


columns_fix=historic_data.filter(regex='fix').columns
print(columns_fix)


# In[15]:


historic_data.loc[historic_data["price_p1_fix"] < 0,"price_p1_fix"] = historic_data["price_p1_fix"].median() 
historic_data.loc[historic_data["price_p2_fix"] < 0,"price_p2_fix"] = historic_data["price_p2_fix"].median() 
historic_data.loc[historic_data["price_p3_fix"] < 0,"price_p3_fix"] = historic_data["price_p3_fix"].median()


# In[16]:


historic_data.head(20)


# In[17]:


mean=historic_data.groupby('id').mean()


# In[18]:


mean


# In[19]:


train=pd.merge(train,mean,on='id')


# In[24]:


train.info()


# In[21]:


train.corr()
date_columns=train.select_dtypes('datetime')
name_columns=['month_activ','month_end','month_modif_prod','month_renewal']
reference_date=dt.datetime(2016,1,1)
for y,x in zip(name_columns,date_columns):
    train[y]=((reference_date-train[x])/np.timedelta64(1,'M')).astype('int')
train.drop(columns=['date_activ','date_end','date_modif_prod','date_renewal'],inplace=True)


# In[23]:


train.fillna(train.mean(),inplace=True)
y=train['churn']
x=train.drop(labels=['id','churn'],axis=1)
print(y)
print(x)


# In[25]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)


# In[26]:


logreg=LogisticRegression(max_iter=1000)


# In[27]:


logreg.fit(x_train,y_train)


# In[28]:


y_pred=logreg.predict(x_test)


# In[29]:


metrics.accuracy_score(y_test,y_pred)


# In[30]:


confusion_matrix=metrics.confusion_matrix(y_test,y_pred)


# In[244]:


confusion_matrix


# In[32]:


logreg.coef_[0]

