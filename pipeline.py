import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import myfunctions_train as ftrain
import myfunctions_test as ftest
sc = MinMaxScaler()

class FeatureHandling():

  def __init__(self):
    self.np = np
    self.pd = pd
    self.sc = sc

  def __delete_useless(self):
    self.dataset = ftrain.delete_unnecessary_features(df=self.dataset,features=['ID','POSTAL_CODE'])
    return self.dataset

  def __imputation_train(self):
    # mean imputation
    self.dataset = ftrain.imputation_mean(df=self.dataset,features=['CREDIT_SCORE'])
    # median imputation
    self.dataset = ftrain.imputation_median(df=self.dataset,features=['ANNUAL_MILEAGE'])
    return self.dataset

  def __encoding_train(self):
    # ordinal encoding
    self.dataset = ftrain.ordinal_encoding(df=self.dataset,features=['AGE', 'DRIVING_EXPERIENCE', 'INCOME'])

    # One-Hot Encoding
    self.dataset = ftrain.one_hot_encoding(df=self.dataset,features=['GENDER', 'RACE', 'EDUCATION', 'VEHICLE_YEAR', 'VEHICLE_TYPE'])
    return self.dataset

  def __scaling(self):
    self.dataset = ftrain.scaling(df=self.dataset)
    return self.dataset

  def __selection(self):
    self.dataset.drop(['VEHICLE_TYPE_sports car', 'RACE_minority', 'EDUCATION_university', 'EDUCATION_none', 'DUIS', 'GENDER_male'],axis=1,inplace=True)
    return self.dataset

  def transform_train(self,dataset):

    self.dataset = dataset.copy()
    self.__delete_useless()
    self.__imputation_train()
    self.__encoding_train()
    # self.__scaling()
    self.__selection()
    return self.dataset




  def __imputation_test(self):
    # mean imputation
    self.dataset = ftest.imputation_mean(df=self.dataset,df_train = self.training_data, features=['CREDIT_SCORE'])
    # median imputation
    self.dataset = ftest.imputation_median(df=self.dataset,df_train = self.training_data, features=['ANNUAL_MILEAGE'])
    return self.dataset

  def __encoding_test(self):
    # ordinal encoding
    self.dataset = ftest.ordinal_encoding(df=self.dataset,df_train = self.training_data,features=['AGE', 'DRIVING_EXPERIENCE', 'INCOME'])

    # One-Hot Encoding
    self.dataset = ftest.one_hot_encoding(df=self.dataset,features=['GENDER', 'RACE', 'EDUCATION', 'VEHICLE_YEAR', 'VEHICLE_TYPE'])
    return self.dataset


  def transform_test(self,testing_data, training_data):

    self.dataset = testing_data.copy()
    self.training_data = training_data.copy()
    self.__delete_useless()
    self.__imputation_test()
    self.__encoding_test()
    # self.__scaling()
    self.__selection()
    return self.dataset

class FeatureHandlingForApp():

  def __init__(self):
    self.np = np
    self.pd = pd
    self.sc = sc

  def __encoding_test(self):
    # ordinal encoding
    self.dataset = ftest.ordinal_encoding(df=self.dataset,df_train = self.training_data,features=['AGE', 'DRIVING_EXPERIENCE', 'INCOME'])
    return self.dataset

  def transform(self,testing_data, training_data):
    self.dataset = testing_data.copy()
    self.training_data = training_data.copy()
    self.__encoding_test()
    return self.dataset
