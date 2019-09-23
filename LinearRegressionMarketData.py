import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy import stats
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

train = pd.read_csv('TrainBM.csv')

#Relabel data that was labeled in different ways
train['Item_Fat_Content'] = train['Item_Fat_Content'].replace({'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'})

#Fill in missing data
train['Item_Weight'] = train['Item_Weight'].fillna(train['Item_Weight'].mean())

#drop feature with ~28% missing data for now and unnecessary data
train = train.drop(['Outlet_Size','Item_Identifier','Outlet_Identifier'], axis = 1)


#Checking to see where there is missing data
missing_data_total = train.isnull().sum().sort_values(ascending = False)
missing_data_percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending = False)
missing_data = pd.concat([missing_data_total, missing_data_percent], axis = 1, keys = ['Total', 'Percent'])


#Normalize the data using a Square Root Transormation
train['Item_Outlet_Sales'] = np.sqrt(train['Item_Outlet_Sales'])
sns.distplot(train['Item_Outlet_Sales'], fit = norm)
#plt.show()

#Encoding data
train_onehot = train.copy()
train_onehot = pd.get_dummies(train_onehot, columns=['Item_Fat_Content','Item_Type','Outlet_Type',
                                                     'Outlet_Location_Type'],
                              prefix = ['Item_Fat_Content','Item_Type','Outlet_Type',
                                                     'Outlet_Location_Type'])

#labeling the training data
X_train = train_onehot.drop(['Item_Outlet_Sales'], axis = 1)
y_train = train_onehot['Item_Outlet_Sales']


#getting out test data.
test_data = pd.read_csv('TestBM.csv')

#Fixing the test data

test_data['Item_Fat_Content'] = test_data['Item_Fat_Content'].replace({'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'})

#Fill in missing data
test_data['Item_Weight'] = test_data['Item_Weight'].fillna(train['Item_Weight'].mean())

#drop feature with ~28% missing data for now and unnecessary data
test_data = test_data.drop(['Outlet_Size'], axis = 1)


test_onehot = test_data.copy()
test_onehot = pd.get_dummies(test_onehot, columns=['Item_Fat_Content','Item_Type','Outlet_Type',
                                                     'Outlet_Location_Type'],
                              prefix = ['Item_Fat_Content','Item_Type','Outlet_Type',
                                                     'Outlet_Location_Type'])

#eliminate unnecessary features
X_test = test_onehot.drop(['Item_Identifier','Outlet_Identifier'], axis = 1)

#print(X_train.info())
#print(test.info())


#training our model for linear regression
from sklearn.linear_model import LinearRegression

linmodel = LinearRegression()
linmodel.fit(X_train,y_train)


#get predictions
predictions = linmodel.predict(X_test)

predictions = (predictions)**2

submission = pd.DataFrame({'Item_Identifier':test_onehot['Item_Identifier'],
                           'Outlet_Identifier':test_onehot['Outlet_Identifier'],
                           'Item_Outlet_Sales':predictions})

#submission.to_csv('BMsubmission3.csv', index = False)


