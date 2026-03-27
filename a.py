import input
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.model_selection import train_test_split  #分割数据集
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error   #模型评价
import warnings

import joblib
E_model = joblib.load('GBRE.joblib')
Ts_model = joblib.load('GBRTs.joblib')
Eb_model = joblib.load('GBREb.joblib')


E=input.E_input()
Eb=input.Eb_input()
Ts=input.Ts_input()

Eb=Eb.values
#df=pd.read_csv('460E.csv')
#df=df.drop(['Unnamed: 0'],axis=1)
#X=df.iloc[:,0:60]
#sc = StandardScaler()
#X = sc.fit_transform(X)
#E = sc.transform(E)
ypreEb=Eb_model.predict(Eb)
print(ypreEb)

E=E.values
#df=pd.read_csv('460E.csv')
#df=df.drop(['Unnamed: 0'],axis=1)
#X=df.iloc[:,0:60]
#sc = StandardScaler()
#X = sc.fit_transform(X)
#E = sc.transform(E)
ypreE=E_model.predict(E)
print(ypreE)


Ts=Ts.values
#df=pd.read_csv('460E.csv')
#df=df.drop(['Unnamed: 0'],axis=1)
#X=df.iloc[:,0:60]
#sc = StandardScaler()
#X = sc.fit_transform(X)
#E = sc.transform(E)
ypreTS=Ts_model.predict(Ts)
print(ypreTS)