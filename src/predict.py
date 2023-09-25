import pandas as pd
import os

import config
import data
import model
import utilities

from model import models

def output_prediction_csv(test_df, predictions) :
    predictions_df = pd.DataFrame(predictions, columns=['survived'])
    
    test_df = test_df['passengerid']
    
    test_df = pd.DataFrame.join(test_df, predictions_df)
    
    path = utilities.get_available_path(config.PREDICTIONS_PATH, 'prediction', 'csv')
    
    test_df[['passengerid', 'survived']].to_csv(path, index=False)
    
if __name__ == '__main__' :
    preprocessor = data.Preprocessor()
    preprocessor.load_transformer(config.TRANSFORMER_PATH)
    
    test_data = data.TestData(preprocessor)    
    
    model = models['random_forest']
    model.load('rf')
    
    predictions = model.predict(test_data.X)
    output_prediction_csv(test_data.test_df, predictions)