from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os

import config

class Model() :
    def __init__(self, directory, model=None) :
        self.model = model
        self.directory = directory
    
    def fit(self, X, y) :
        self.model.fit(X, y)
    
    def predict(self, X) :
        return self.model.predict(X)
    
    def save(self, model_name) :
        path = os.path.join(self.directory, '{}.joblib'.format(model_name))
        joblib.dump(self.model, path)
    
    def load(self, model_name) :
        path = os.path.join(self.directory, '{}.joblib'.format(model_name))
        self.model = joblib.load(path)
    
    def cross_val_score(self, X, y, cv=5) :
        scores = cross_val_score(estimator=self.model, X=X, y=y, cv=cv)
        
        return scores
    
models = {
    'random_forest': Model(config.RANDOM_FOREST_PATH)
}