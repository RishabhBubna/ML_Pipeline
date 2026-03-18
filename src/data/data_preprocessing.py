## Importing necessary libraries:
import os

## Data manipulation
import numpy as np
import pandas as pd

## Machine learning
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

## Logging
import logging

## yaml
import yaml

## save file
import joblib
import json

## Setting seed 
np.random.seed(26)

# ## logging configuration
# logger = logging.getLogger("data_preprocessing")
# logger.setLevel(logging.DEBUG)

# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.DEBUG)

# file_handler = logging.FileHandler("preprocessing_errors.log")
# file_handler.setLevel(logging.ERROR)

# formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# console_handler.setFormatter(formatter)
# file_handler.setFormatter(formatter)

# logger.addHandler(console_handler)
# logger.addHandler(file_handler)

from utils import load_params, setup_logger

logger = setup_logger("data_preprocessing","preprocessing_error.log")


# def load_params(params_path: str)-> dict:
#     '''Load the parameters from the params.yaml file'''
#     try:
#         with open(params_path, "r") as f:
#             params = yaml.safe_load(f)
#         logger.debug("Parameters were loaded from %s", params_path)
#         return params
#     except FileNotFoundError:
#         logger.error("File not found: %s", params_path)
#         raise
#     except yaml.YAMLError as e:
#         logger.error("YAML error: %s", e)
#         raise
#     except Exception as e:
#         logger.error("Unexpected error: %s", e)
#         raise

def data_splitting(df: pd.DataFrame, test_size: float)-> tuple:
    '''Chronologically splitting the data into train and test set'''
    try:
        df = df.sort_values(by = "TransactionDT") # sorting according to TransactionDT
        X = df.drop(columns=["isFraud","TransactionDT"]) 
        y = df["isFraud"]
        train_X ,test_X, train_y, test_y = train_test_split(X, y, test_size= test_size, random_state=26, shuffle = False)
        train_X_normal = train_X[train_y == 0]
        logger.debug("Data split successfully")
        return train_X_normal, test_X, test_y
    except Exception as e:
        logger.error("Data split unsuccessful: %s", e)
        raise

def iso_pipeline(df_train: pd.DataFrame, df_test: pd.DataFrame)-> list:
    '''Define the isolation forest data pipeline and save the data as numpy array'''
    try:
        num_col_Iso = df_train.select_dtypes(include = ["number"]).columns.tolist()
        cat_col_Iso = df_train.select_dtypes(include=["object"]).columns.tolist()
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        preprocessor_Iso = ColumnTransformer(transformers=[
            ('num', num_pipeline, num_col_Iso),
            ('cat', cat_pipeline, cat_col_Iso)
        ])
        
        train_processed_Iso = preprocessor_Iso.fit_transform(df_train)
        pipeline_path =  os.path.join(os.path.dirname(os.path.abspath(__file__)),"../model/pipeline")

        os.makedirs(pipeline_path, exist_ok=True)

        joblib.dump(preprocessor_Iso,os.path.join(pipeline_path,"transform_rule_Iso.pkl"))

        train_processed_Iso = train_processed_Iso.astype("float32")
        test_processed_Iso = preprocessor_Iso.transform(df_test).astype("float32")

        preprocessed_path =  os.path.join(os.path.dirname(os.path.abspath(__file__)),"../processedData/preprocessed")
        os.makedirs(preprocessed_path, exist_ok=True)
        
        np.save(os.path.join(preprocessed_path,"trainset_Iso.npy") ,train_processed_Iso)
        np.save(os.path.join(preprocessed_path,"testset_Iso.npy" ),test_processed_Iso)

        feature_list_Iso = preprocessor_Iso.get_feature_names_out()
        feature_list_Iso = feature_list_Iso.tolist()

        logger.debug("Data for isolation forest successfully processed and saved")

        return feature_list_Iso
    except Exception as e:
        logger.error("isolation forest data processing failed: %s", e)
        raise

def VAE_pipeline(df_train: pd.DataFrame, df_test: pd.DataFrame)-> list:
    '''Define the VAE data pipeline and save the data as numpy array'''
    try:
        v_col = [c for c in df_train.columns if c.startswith("V")]
        train_X_normal_VAE = df_train.drop(columns=v_col)
        test_X_VAE = df_test.drop(columns=v_col)

        num_col_VAE = train_X_normal_VAE.select_dtypes(include = ["number"]).columns.tolist()
        cat_col_VAE = train_X_normal_VAE.select_dtypes(include=["object"]).columns.tolist()
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        preprocessor_VAE = ColumnTransformer(transformers=[
            ('num', num_pipeline, num_col_VAE),
            ('cat', cat_pipeline, cat_col_VAE)
        ])
        
        train_processed_VAE = preprocessor_VAE.fit_transform(train_X_normal_VAE)

        pipeline_path =  os.path.join(os.path.dirname(os.path.abspath(__file__)),"../model/pipeline")

        os.makedirs(pipeline_path, exist_ok=True)
        joblib.dump(preprocessor_VAE,os.path.join(pipeline_path,"transform_rule_VAE.pkl"))

        preprocessed_path =  os.path.join(os.path.dirname(os.path.abspath(__file__)),"../processedData/preprocessed")
        os.makedirs(preprocessed_path, exist_ok=True)
        train_processed_VAE = train_processed_VAE.astype("float32")
        test_processed_VAE = preprocessor_VAE.transform(test_X_VAE).astype("float32")

        np.save(os.path.join(preprocessed_path,"trainset_VAE.npy") ,train_processed_VAE)
        np.save(os.path.join(preprocessed_path,"testset_VAE.npy") ,test_processed_VAE)

        feature_list_VAE = preprocessor_VAE.get_feature_names_out()
        feature_list_VAE = feature_list_VAE.tolist()
        logger.debug("Data for VAE successfully processed and saved")

        return feature_list_VAE
    except Exception as e:
        logger.error("VAE data processing failed: %s", e)
        raise

def main():

    try:

        logger.debug("Starting the preprocessing of data")

        df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../processedData/raw/full_dataset.csv"))

        logger.debug("Data loaded successfully")

        params = load_params(params_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"../params.yaml"))
        test_size = params["data_preprocessing"]["test_size"]

        df_train, df_test, df_y = data_splitting(df, test_size)

        preprocessed_path =  os.path.join(os.path.dirname(os.path.abspath(__file__)),"../processedData/preprocessed")
        os.makedirs(preprocessed_path, exist_ok=True)
        np.save(os.path.join(preprocessed_path,"testset_label.npy") ,df_y)

        logger.debug("Data split and test Labels saved")

        name_iso = iso_pipeline(df_train, df_test)
        name_VAE = VAE_pipeline(df_train, df_test)

        feature_dic = {
            "Model_Name" : ["VAE","Iso"],
            "No. of rows" : [len(name_VAE),len(name_iso)],
            "feature_name" : [name_VAE,name_iso]
                    }
        
        json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"../Metadata")

        os.makedirs(json_path, exist_ok=True)
        with open(os.path.join(json_path,"feature_name.json"), "w") as f:
            json.dump(feature_dic,f)

        logger.debug("Feature names saved and all preprocessing done")

    except Exception as e:
        logger.error("data processing failed: %s", e)
        raise

if __name__ == "__main__":
    main()