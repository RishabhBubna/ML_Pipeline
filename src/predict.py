## Importing necessary libraries:
import os

## Data manipulation
import numpy as np
import pandas as pd

## deep learning
import torch
from torch.utils.data import DataLoader, TensorDataset

## save file
import joblib
import json

## absolute path for data
from config import TEST_TRANSACTION_PATH, TEST_IDENTITY_PATH
from utils import setup_logger
from data.data_ingestion import (extract_temporal_features,clean_id30,clean_id31,bin_resolution,clean_device_info,merge_df)
from model.model_evaluation import (evaluate_vae,evaluate_iso)
from model.VAE import MyVAE

logger = setup_logger("prediction", "prediction_error.log")

def load_data(test_data_path: str, column_list)-> pd.DataFrame:
    '''loads data with relevant columns'''
    try:
        available_cols = pd.read_csv(test_data_path, nrows=0).columns.tolist()
        cols_to_load = [c for c in column_list if c in available_cols]
        df = pd.read_csv(test_data_path, usecols=cols_to_load, nrows=20)
        logger.debug("data loaded from: %s", test_data_path)
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse CSV file: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error encountered while loading data: %s", e)
        raise

def align_columns(df: pd.DataFrame, column_list: list) -> pd.DataFrame:
    '''Align dataframe columns to match training columns'''
    try:

        for col in column_list:
            if col not in df.columns:
                df[col] = np.nan
                logger.debug("Missing column added as NaN: %s", col)
        
        df = df[column_list]
        logger.debug("Columns aligned, shape: %s", df.shape)
        return df
    except Exception as e:
        logger.error("Column alignment failed: %s", e)
        raise

def load_models(vae_path: str, iso_path: str, input_dim: int, z_dim: int) -> tuple:
    '''Load VAE and ISO models from local paths'''
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        vae_model = MyVAE(input_dim, z_dim).to(device)
        vae_model.load_state_dict(torch.load(vae_path, map_location=device))
        vae_model.eval()
        logger.debug("VAE model loaded from: %s", vae_path)

        iso_model = joblib.load(iso_path)
        logger.debug("Isolation forest loaded from: %s", iso_path)

        return vae_model, iso_model, device
    except Exception as e:
        logger.error("Failed to load models: %s", e)
        raise


def load_pipeline(vae_pipeline_path: str, iso_pipeline_path: str) -> tuple:
    '''Load VAE and ISO transform rules from local paths'''
    try:
        vae_pipeline = joblib.load(vae_pipeline_path)
        logger.debug("VAE pipeline loaded from: %s", vae_pipeline_path)

        iso_pipeline = joblib.load(iso_pipeline_path)
        logger.debug("ISO pipeline loaded from: %s", iso_pipeline_path)

        return vae_pipeline, iso_pipeline
    except Exception as e:
        logger.error("Failed to load pipelines: %s", e)
        raise

def main():
    try:
        root_dir = os.path.dirname(os.path.abspath(__file__))

        # Load params
        vae_w = 0.9
        iso_w = 0.1
        threshold = 0.1993
        z_dim = 3
        batch_size = 2048

        # Load column lists
        with open(os.path.join(root_dir, "Metadata/column_name.json"), "r") as f:
            columns = json.load(f)

        # Load raw data
        test_transaction_path = TEST_TRANSACTION_PATH
        test_identity_path = TEST_IDENTITY_PATH

        transaction_list = columns["transaction_list"]
        identity_list = columns["Identity_list"]

        df_t = load_data(test_transaction_path, transaction_list)
        df_i = load_data(test_identity_path, identity_list)

        # Clean
        df_t = extract_temporal_features(df_t)
        transaction_list.extend(["hour","day_of_week"])
        # Align columns
        df_t = align_columns(df_t, transaction_list)
        df_i = align_columns(df_i, identity_list)

        df_i["id_30"] = df_i["id_30"].apply(clean_id30)
        df_i["id_31"] = df_i["id_31"].apply(clean_id31)
        df_i["id_33"] = df_i["id_33"].apply(bin_resolution)
        df_i["DeviceInfo"] = df_i["DeviceInfo"].apply(clean_device_info)

        # Merge
        df = merge_df(df_t, df_i)

        # Log transform
        for col in columns["log_list"]:
            df[col] = np.log1p(df[col])
        logger.debug("log-transform done successfully")

        # Split for VAE and ISO
        v_cols = [c for c in df.columns if c.startswith("V")]
        df_vae = df.drop(columns=v_cols)
        df_iso = df.copy()

        # Load pipelines
        vae_pipeline, iso_pipeline = load_pipeline(
            vae_pipeline_path=os.path.join(root_dir, "model/pipeline/transform_rule_VAE.pkl"),
            iso_pipeline_path=os.path.join(root_dir, "model/pipeline/transform_rule_Iso.pkl")
        )

        # Apply transform rules
        df_vae_transformed = vae_pipeline.transform(df_vae).astype("float32")
        df_iso_transformed = iso_pipeline.transform(df_iso).astype("float32")

        # Infer input_dim and load models
        input_dim = df_vae_transformed.shape[1]
        vae_model, iso_model, device = load_models(
            vae_path=os.path.join(root_dir, "model/best_vae.pt"),
            iso_path=os.path.join(root_dir, "model/iso_forest.pkl"),
            input_dim=input_dim,
            z_dim=z_dim
        )

        # Get scores
        tensor = torch.tensor(df_vae_transformed, dtype=torch.float32)
        test_dataloader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=False)
        vae_score = evaluate_vae(model=vae_model, test_dataloader=test_dataloader, device=device)
        iso_score = evaluate_iso(model=iso_model, test_X=df_iso_transformed)

        # Ensemble
        ensemble_scores = vae_w * vae_score + iso_w * iso_score
        predictions = (ensemble_scores >= threshold).astype(int)

        # Save results
        results = pd.DataFrame({
            "ensemble_score": ensemble_scores,
            "prediction": predictions
        })

        results_path = os.path.join(root_dir, "results/predictions.csv")
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        results.to_csv(results_path, index=False)
        logger.debug("Predictions saved to: %s", results_path)

    except Exception as e:
        logger.error("Prediction pipeline failed: %s", e)
        raise


if __name__ == "__main__":
    main()