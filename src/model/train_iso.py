## Importing necessary libraries:
import os

## Data manipulation
import numpy as np
import pandas as pd

## Isolation forest
from sklearn.ensemble import IsolationForest

## File saving
import joblib

from utils import load_params, setup_logger

logger = setup_logger("training_ISO", "ISO_error.log")

def data_loader(file_path: str)-> float:
    '''Load data for training the Isolation forest'''
    try:

        train_X = np.load(file_path)

        logger.debug("Data loaded for isolation forest, shape of data: %s", train_X.shape)
        return train_X
    except Exception as e:
        logger.error("data loader for isolation forest failed: %s", e)
        raise

def train_ISO(params: dict)-> None:
    '''Train the Isolation forest model and save it'''
    try:

        n_estimator = params["train_iso"]["n_estimator"]
        contamination = params["train_iso"]["contamination"]
        max_feature = params["train_iso"]["max_feature"]
        random_state = params["train_iso"]["random_state"]

        preprocessed_path =  os.path.join(os.path.dirname(os.path.abspath(__file__)),"../processedData/preprocessed")
        train_data_file_path = os.path.join(preprocessed_path, "trainset_Iso.npy")
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"iso_forest.pkl")

        train_X = data_loader(train_data_file_path)

        iso_forest = IsolationForest(n_estimators=n_estimator, contamination=contamination, max_features=max_feature,
                                         random_state= random_state, n_jobs=-1)
        logger.debug("Isolation forest model initialised.")

        iso_forest.fit(train_X)

        
        joblib.dump(iso_forest, save_path)

        logger.debug("Isolation forest model trained and saved.")

    except Exception as e:
        logger.error("Isolation forest training failed: %s", e)
        raise

def main():
    try:
        params = load_params(params_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"../params.yaml"))

        train_ISO(params)

    except Exception as e:
        logger.error("Training failed: %s", e)
        raise 

if __name__ == "__main__":
    main()