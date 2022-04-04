# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 13:42:44 2022

@author: 47700100
"""
#%%
from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score,plot_confusion_matrix
from numpy import linalg as LA
from sklearn.preprocessing import MinMaxScaler
# Import label encoder
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import time
import pickle
#%%
def generate_model_report(y_actual, y_predicted):
    print("Accuracy : " ,"{:.4f}".format(accuracy_score(y_actual, y_predicted)))     
    auc = roc_auc_score(y_actual, y_predicted)
    print("AUC : ", "{:.4f}".format(auc))
#%%
data=pd.read_csv('data/adult.csv')
data_org=data.copy()

numerical_list=[]
categorical_list=[]
for i in data.columns.tolist():
    if data[i].dtype=='object':
        categorical_list.append(i)
    else:
        numerical_list.append(i)
print('Number of categorical features:', str(len(categorical_list)))
print('Number of numerical features:', str(len(numerical_list)))

label_encoder = preprocessing.LabelEncoder()
for col in categorical_list:
    data[col]=label_encoder.fit_transform(data[col])
    
#%%    
columns=['age', 'workclass', 'fnlwgt', 'education', 'educational-num',
       'marital-status', 'occupation', 'relationship', 'race', 'gender',
       'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
       'income']
data_model=data[columns] 
data1=data_model
X = data1.iloc[:, :-1]
y = data1.iloc[:, -1]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 12)  
X_train.to_csv('data/X_train.csv',index=False)  
X_test.to_csv('data/X_test.csv',index=False)  

model = RandomForestClassifier(n_estimators=10)
model.fit(X_train, y_train)
print("Training Report................ ")
generate_model_report(y_train,model.predict(X_train))
print("Testing Report................ ")
y_pred = model.predict(X_test)
generate_model_report(y_test,y_pred)
feat_importances_rf = pd.Series(model.feature_importances_, index=X_train.columns)
important_features_rf=feat_importances_rf.nlargest(50)
print('Feature Importance........') 
print(important_features_rf) 

    

