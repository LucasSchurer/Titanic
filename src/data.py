from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import joblib
import numpy as np
import pandas as pd
import os

import config

class Features() :
    def __init__(self) :
        self.auxiliary_features = ['name']
        self.numerical_features = ['age', 'sibsp', 'parch', 'fare']
        self.categorical_features = ['pclass', 'sex', 'embarked']    
            
class AgeTransformer(BaseEstimator, TransformerMixin) :
    def fit(self, X, y=None) :
        return self
    
    def transform(self, X):        
        young_male_df = X.loc[X['name'].str.contains('Master.')]
        male_df = X.loc[(X['sex'] == 'male') & (~X['name'].str.contains('Master.'))]
        female_df = X.loc[(X['sex'] == 'female')]
        
        young_male_age_mean = young_male_df['age'].mean()
        male_age_mean = male_df['age'].mean()
        female_age_mean = female_df['age'].mean()
        
        X['age'] = X.apply(
            lambda row : 
                row['age'] if not np.isnan(row['age']) 
                else self.__get_row_age_by_sex(row, 
                                             male_age_mean, young_male_age_mean, 
                                             female_age_mean),
            axis=1
        )
        
        return X[['age']]
    
    def __get_row_age_by_sex(self, row, male_age_mean, young_male_age_mean, female_age_mean) :
        if row['sex'] == 'female' :
            return female_age_mean
    
        if ('Master.') in row['name'] :
            return young_male_age_mean 
        else :
            return male_age_mean
        
    def get_feature_names_out(self, feature_names) :
        return ['age']

class Preprocessor() :
    def __init__(self) :
        self.transformer = None

    def build_transformer(self, features, scaler) :
        numerical_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='median'))
            ]
        )
        
        if scaler == 'standard' :
            numerical_transformer.steps.append(['scaler', StandardScaler()])
        if scaler == 'minmax' :
            numerical_transformer.steps.append(['scaler', MinMaxScaler()])

        categorical_transformer = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
                ]
            )
        
        age_transformer = AgeTransformer()

        age_transformer_features = ['age', 'sex', 'name']
        numerical_transformer_features = [x for x in features.numerical_features if x not in age_transformer_features]

        transformer = ColumnTransformer( [
            ('age_transformer', age_transformer, age_transformer_features),
            ('numerical_transformer', numerical_transformer, numerical_transformer_features),
            ('categorical_transformer', categorical_transformer, features.categorical_features),
        ], verbose_feature_names_out=False)
        
        self.transformer = transformer
        
        return self.transformer
        
    def save_transformer(self, path) :
        if self.transformer is not None :
            joblib.dump(self.transformer, path)    
            
    def load_transformer(self, path) :
        if os.path.exists(path) :
            self.transformer = joblib.load(path)

def read_csv(type = 'train') :
        file_name = '{}.{}'.format(type, 'csv')
        path = os.path.join(config.DATA_PATH, file_name)
        
        data = pd.read_csv(path)
        
        data.columns = data.columns.str.lower()
    
        return data

class TrainingData() :
    def __init__(self, preprocessor_scaler = 'standard') :
        self.features = Features()
        self.X, self.y = self.__get_X_y()
        
        self.preprocessor = Preprocessor()
        self.preprocessor.build_transformer(self.features, preprocessor_scaler)
        self.preprocessor.transformer.fit(self.X, self.y)
        
        self.X = self.preprocessor.transformer.transform(self.X)

    def __get_X_y(self) :
        data = read_csv('train')
        
        X = data.drop(columns=['survived'])
        X = X[self.features.numerical_features + self.features.categorical_features + self.features.auxiliary_features]

        y = data['survived']
        
        return X, y

class TestData() :
    def __init__(self, preprocessor : Preprocessor) :
        self.X = self.__get_X()
        self.test_df = read_csv('test')
        self.preprocessor = preprocessor
        
        self.X = preprocessor.transformer.transform(self.X)
        
    def __get_X(self) :
        X = read_csv('test')
        
        return X