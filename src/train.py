from sklearn.model_selection import cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import joblib
import os

import data
import model
import config
import utilities

def get_search_results_overview(search_results, thresholds = None, order_by='mean_test_score', ascending=True) :    
        
    # display('Best estimator: {}'.format(search_results.best_params_))
    # display('Best score: {}'.format(search_results.best_score_))

    if thresholds != None :
        df = pd.DataFrame(search_results.cv_results_)             
    
        for key in thresholds :
            if len(thresholds[key]) == 1 :
                df = df.loc[df[key] >= thresholds[key][0]]
            else :    
                df = df.loc[df[key] <= thresholds[key][0] if thresholds[key][1] == 'max' else df[key] >= thresholds[key][0]]
            
        df = df.sort_values(order_by, ascending=ascending)
        
        param_columns = []
        
        for column in df.columns :
            if 'param' in column :
                param_columns.append(column)
        
        score_columns = ['mean_test_score', 'std_test_score']
        
        df = df[score_columns + param_columns]
        
        return df

def randomized_search(model : model.Model, params, X_train, y_train, n_iter=100, 
                      cv=5, verbose=3, random_state=None, 
                      n_jobs=-1, save_cv_results=True, save_best_estimator=True) :
    
    search = RandomizedSearchCV(estimator=model.model, param_distributions=params, 
                                n_iter=n_iter, cv=cv, verbose=verbose, 
                                random_state=random_state, n_jobs=n_jobs)
    
    search.fit(X_train, y_train)
    
    if save_cv_results :
        save_search(search, 'random search', model.directory, save_best_estimator)

    return search

def grid_search(model : model.Model, params, X_train, y_train, scoring='accuracy', 
                cv=5, verbose=3, n_jobs=-1, 
                save_cv_results=True, save_best_estimator=True) :
    
    search = GridSearchCV(estimator=model.model, param_grid=params, scoring=scoring, 
                          cv=cv, verbose=verbose, n_jobs=n_jobs)
    
    search.fit(X_train, y_train)
        
    if save_cv_results :
        save_search(search, 'grid search', model.directory, save_best_estimator)
        
    return search

def save_search(search, search_type, directory, save_best_estimator = True) :
    current_time = utilities.get_current_time()
    
    filename = '{} - {}'.format(search_type, current_time)
    search_directory = os.path.join(directory, 'search_results')
    
    path = utilities.get_available_path(search_directory, filename, 'csv')
    
    pd.DataFrame(search.cv_results_).to_csv(path)
    
    if save_best_estimator :
        score = search.best_score_
        filename = '{}'.format(score)
        model_directory = directory
        
        path = utilities.get_available_path(model_directory, filename, 'joblib')

        joblib.dump(search.best_estimator_, path)

def rf_randomized_search_params(print_params = False) :
    n_estimators = [int(x) for x in np.linspace(100, 200, num=20)]
    max_features = ['sqrt', 'log2']
    max_depth = [int(x) for x in np.linspace(10, 100, num=20)]
    max_depth.append(None)
    min_samples_split = np.arange(2, 10, 2)
    min_samples_leaf = np.arange(2, 20, 2)
    bootstrap = [True]
    
    params = {
        'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'bootstrap': bootstrap
    }
    
    if print_params :
        for param in params :
            print('{}: {}'.format(param, params[param]))
    
    return params
    
def rf_grid_search_params(print_params = False) :
    # n_estimators = [int(x) for x in np.linspace(100, 200, num=20)]
    # max_features = ['sqrt', 'log2']
    # max_depth = [int(x) for x in np.linspace(10, 100, num=20)]
    # max_depth.append(None)
    # min_samples_split = np.arange(2, 10, 2)
    # min_samples_leaf = np.arange(2, 20, 2)
    # bootstrap = [True]
    
    n_estimators = [100, 200]
    max_features = ['sqrt', 'log2']
    max_depth = [2, 4, 6]
    max_depth.append(None)
    min_samples_split = [2, 4]
    min_samples_leaf = [2, 4, 6]
    bootstrap = [True]
    
    params = {
        'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'bootstrap': bootstrap
    }
    
    if print_params :
        for param in params :
            print('{}: {}'.format(param, params[param]))
    
    return params

def dt_grid_search_params(print_params = False) :
    splitter = ['best']
    max_depth = [int(x) for x in np.linspace(5, 100, num=15)]
    max_depth.append(None)
    min_samples_split = [4, 2]
    min_samples_leaf = np.arange(2, 10, 2)
    min_weight_fraction_leaf = [x for x in np.linspace(0, 0.1, 150)]
    max_features = ['log2', None]
    max_leaf_nodes = [int(x) for x in np.linspace(50, 600, num=30)]
    max_leaf_nodes.append(None)
    min_impurity_decrease = [0]

    params = {
        'splitter': splitter,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'min_weight_fraction_leaf': min_weight_fraction_leaf,
        'max_features': max_features,
        'max_leaf_nodes': max_leaf_nodes,
        'min_impurity_decrease': min_impurity_decrease
    }
    
    if print_params :
        for param in params :
            print('{}: {}'.format(param, params[param]))
    
    return params
    
if __name__ == '__main__' :
    training_data = data.TrainingData()
    
    rf = model.Model(directory=config.RANDOM_FOREST_PATH)
    rf.model = RandomForestClassifier()
    params = rf_grid_search_params(False)
    
    search_results = grid_search(rf, params, training_data.X, training_data.y)