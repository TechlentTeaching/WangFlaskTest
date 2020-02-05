#!/usr/bin/env python
# coding: utf-8

# ## try to print prediction at the end in Terminal

# In[1]:


import pandas as pd
import numpy as np
import random
from math import sqrt
from matplotlib import pyplot as plt


# In[2]:


from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline# 3. Analyze the Data## 3.1 Try Linear Regression as a Start

from sklearn import preprocessing


# In[3]:


## Code to keep (3) ====> To process both numerical and categorical data

from sklearn import preprocessing

def all_feature(X):
    df = pd.DataFrame()
    #df = X  # May need to delete this line if only want to keep transfered numerical data
    #df['PRICE'] = X['PRICE'].map(lambda x: float(x.strip('$').replace(',', '')) if type(x) == str else x)
    df['BEDS'] = X['BEDS'].map(lambda x: float(x.strip('$').strip('k').replace(',', '')) if type(x) == str else x)
    df['BATHS'] = X['BATHS'].map(lambda x: float(x.strip('$').strip('k').replace(',', '')) if type(x) == str else x)
    df['SQUARE FEET'] = X['SQUARE FEET'].map(lambda x: float(x.strip('$').strip('k').replace(',', '')) if type(x) == str else x)
    df['LOT SIZE'] = X['LOT SIZE'].map(lambda x: 0 if type(x) != float else x)
    df['YEAR BUILT'] = X['YEAR BUILT'].map(lambda x: float(x.strip('$').strip('k').replace(',', '')) if type(x) == str else x)
    df## 1.3 Clean up numerical data and visualization['DAYS ON MARKET'] = X['DAYS ON MARKET'].map(lambda x: float(x.strip('$').strip('k').replace(',', '')) if type(x) == str else x)
    df['HOA/MONTH'] =  X['HOA/MONTH'].map(lambda x: 0 if type(x) != float else x)
    #df['views'] = X['views'].map(lambda x: float(x.strip('$').strip('k').replace(',', '')) if type(x) == str else x)
    df['PROPERTY TYPE'] = X['PROPERTY TYPE'].map(lambda x: "No" if x == '' or pd.isnull(x) or type(x) != str else x)
    df['floor'] = X['floor'].map(lambda x: "No" if x == '' or pd.isnull(x) or type(x) != str else x)

## Below city/location could be complementary to lat/long info, we only need to keep one
    df['CITY'] = X['CITY'].map(lambda x: "No" if x == '' or pd.isnull(x) or type(x) != str else x)
    df['LOCATION'] = X['LOCATION'].map(lambda x: "No" if x == '' or pd.isnull(x) or type(x) != str else x)
    return df



def numerical_feature(X):
    df = pd.DataFrame()
    #df = X  # May need to delete this line if only want to keep transfered numerical data
    df['PRICE'] = X['PRICE'].map(lambda x: float(x.strip('$').replace(',', '')) if type(x) == str else x)
    df['BEDS'] = X['BEDS'].map(lambda x: float(x.strip('$').strip('k').replace(',', '')) if type(x) == str else x)
    df['BATHS'] = X['BATHS'].map(lambda x: float(x.strip('$').strip('k').replace(',', '')) if type(x) == str else x)
    df['SQUARE FEET'] = X['SQUARE FEET'].map(lambda x: float(x.strip('$').strip('k').replace(',', '')) if type(x) == str else x)
    df['LOT SIZE'] = X['LOT SIZE'].map(lambda x: np.nan if type(x) != float else x)
    df['YEAR BUILT'] = X['YEAR BUILT'].map(lambda x: float(x.strip('$').strip('k').replace(',', '')) if type(x) == str else x)
    df## 1.3 Clean up numerical data and visualization['DAYS ON MARKET'] = X['DAYS ON MARKET'].map(lambda x: float(x.strip('$').strip('k').replace(',', '')) if type(x) == str else x)
    df['HOA/MONTH'] =  X['HOA/MONTH'].map(lambda x: np.nan if type(x) != float else x)
    df['LATITUDE'] = X['LATITUDE'] .map(lambda x: float(x.strip('$').strip('k').replace(',', '')) if type(x) == str else x)
    df['LONGITUDE'] = X['LONGITUDE'].map(lambda x: float(x.strip('$').strip('k').replace(',', '')) if type(x) == str else x)
    return df

class Create_model(object):
    def __init__(self):
        self.modelRFT = RandomForestRegressor(max_depth = 20, n_estimators = 500)
        #print(self.modelRFT)
    def fitX(self, X, y=None):
        # for numerical data processing.....
        X_Nmbr = all_feature(X)
        self.mean = X_Nmbr.median() 
        # for categorical data processing.....
        tempList=[]
        self.ctgrColList = ['PROPERTY TYPE', 'floor','CITY','LOCATION'] #,'parking','bsmt']#'STATE OR PROVINCE', 
        return #set(tempList)
                        
    def transformX(self, X, y=None):
        # for numerical data processing.....
        X_Nmbr = all_feature(X).fillna(self.mean)
        
        for ctgrCol in self.ctgrColList:  
            X_Nmbr[ctgrCol] = preprocessing.LabelEncoder().fit_transform(X_Nmbr[ctgrCol])
        
        return X_Nmbr #pd.concat([X_Nmbr, pd.DataFrame(res,index = X_Nmbr.index)], axis=1)     

    def fit_transformX(self, X, y=None):
        self.fitX(X)
        return self.transformX(X)
    
    def fit(self, X, y):
        X_trs = self.fit_transformX(X)
        #self.modelRFT = RandomForestRegressor(max_depth = 20, n_estimators = 500)
        self.modelRFT.fit(X_trs, y)
        return #self.transform(X)

    def predict(self, X, y=None):
        X_trs = self.transformX(X)
        return self.modelRFT.predict(X_trs)


# In[4]:


X_train = pd.read_csv('X_train.csv')
X_train.set_index('id', drop=True, append=False, inplace=True, verify_integrity=False)
y_train = pd.read_csv('y_train.csv', index_col = 0, header = None)
print(y_train)
#y_train.set_index(0, drop=True, append=False, inplace=True, verify_integrity=False)
print(y_train.shape)
print(X_train.shape)
print(y_train.tail())
X_train.tail()
print(y_train.index)
X_train.index
X_train.columns


# In[5]:


## ** Below Codes can be deleted ***
#X_test = pd.read_csv('../data/NYdata/X_test.csv')
#X_test.set_index('id', drop=True, append=False, inplace=True, verify_integrity=False)
#y_test = pd.read_csv('../data/NYdata/y_test.csv', index_col = 0, header = None)
## ** Above Codes can be deleted 


# In[ ]:





# In[6]:


model = Create_model() #wrapper for transformer & RFT
model.fit(X_train, y_train)

## ** Below Codes can be deleted ***
#TestRMSE = sqrt(mean_squared_error(y_test, model.predict(X_test)))
#TestR2 = r2_score(y_test, model.predict(X_test))
#TestMAE = mean_absolute_error(y_test, model.predict(X_test))
#TrainRMSE = (sqrt(mean_squared_error(y_train, model.predict(X_train))))
#TrainR2 = (r2_score(y_train, model.predict(X_train)))
#TrainMAE = (mean_absolute_error(y_train, model.predict(X_train)))

#print('train//test RMSE:', TrainRMSE,' || ', TestRMSE)
#print('train//test  MAE:', TrainMAE,' || ', TestMAE)
#print('train//test   R2:', TrainR2, ' || ',TestR2)
## ** Above Codes can be deleted 


# In[7]:


X_train1 = X_train[X_train.index == 'NY_10025']
print('{0:.0f}'.format(model.predict(X_train1)[0]))
X_train1.columns


# In[8]:


import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression

#df = pd.read_csv("haberman.csv")
#X = X_train
#y = y_train
# 1: died within five years, 0: survived 5+ years

#model.fit(X_train, y)
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)


# In[ ]:




