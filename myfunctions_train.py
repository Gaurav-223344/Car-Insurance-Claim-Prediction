import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
enc_oe = OrdinalEncoder()
sc_s = StandardScaler()

def delete_unnecessary_features(df,features):
  df.drop(columns=features,axis=1,inplace=True)
  return df


#  imputations

def imputation_mean(df,features):
  for feature in features:
    df[feature].fillna(df[feature].mean(),inplace=True)
  return df

def imputation_median(df,features):
  for feature in features:
    df[feature].fillna(df[feature].median(),inplace=True)
  return df

def imputation_frequency(df,features):
  for feature in features:
    df[feature].fillna(df[feature].mode()[0],inplace=True)
  return df

#  encoding

def ordinal_encoding(df,features):
  df[features] = enc_oe.fit_transform(df[features])
  return df

def one_hot_encoding(df,features):
  dummy = pd.get_dummies(df[features],drop_first=True)
  data = df.join([dummy])
  data.drop(columns=features,axis=1,inplace=True)
  return data

#  scaling

def scaling(df):
  temp = sc_s.fit_transform(df)
  clms = df.columns
  df = pd.DataFrame(temp,columns=clms)
  return df



  




