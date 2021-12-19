import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
enc_oe = OrdinalEncoder()


#  imputations

def imputation_mean(df,df_train,features):
  for feature in features:
    df[feature].fillna(df_train[feature].mean(),inplace=True)
  return df

def imputation_median(df,df_train,features):
  for feature in features:
    df[feature].fillna(df_train[feature].median(),inplace=True)
  return df

def imputation_frequency(df,df_train,features):
  for feature in features:
    df[feature].fillna(df_train[feature].mode()[0],inplace=True)
  return df

#  encoding

def ordinal_encoding(df,df_train,features):
  enc_oe.fit(df_train[features])
  df[features] = enc_oe.transform(df[features])
  return df

def one_hot_encoding(df,features):
  dummy = pd.get_dummies(df[features],drop_first=True)
  data = df.join([dummy])
  data.drop(columns=features,axis=1,inplace=True)
  return data

