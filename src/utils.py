import os 
import sys
import numpy as np
import pandas as pd
from src.exceptions import CustomException
from src.logger import logging
import dill
from sklearn.metrics import r2_score


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
        logging.info("File saved")
    except Exception as e:
        raise CustomException(e, sys)
    
    
def evaluate_models(xtrain, ytrain, xtest, ytest, models):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(xtrain, ytrain)
            y_train_pred = model.predict(xtrain)
            y_test_pred = model.predict(xtest)
            train_model_score = r2_score(ytrain, y_train_pred)
            test_model_score = r2_score(ytest, y_test_pred)
            report[list(models.keys())[i]] = test_model_score
        return report
    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)