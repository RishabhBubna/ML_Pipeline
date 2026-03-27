import os

## Data manipulation
import numpy as np
import pandas as pd

## deep learning
import torch
from torch.utils.data import DataLoader, TensorDataset

## File saving
import joblib
import json

from data.data_ingestion import (extract_temporal_features,clean_id30,clean_id31,bin_resolution,clean_device_info,merge_df)
from model.model_evaluation import (evaluate_vae,evaluate_iso)
from model.VAE import MyVAE


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

VAE_MODEL_PATH = os.path.join(ROOT_DIR, "model/best_vae.pt")
ISO_MODEL_PATH = os.path.join(ROOT_DIR, "model/iso_forest.pkl")
VAE_PIPELINE_PATH = os.path.join(ROOT_DIR, "model/pipeline/transform_rule_VAE.pkl")
ISO_PIPELINE_PATH = os.path.join(ROOT_DIR, "model/pipeline/transform_rule_Iso.pkl")
COLUMN_METADATA_PATH = os.path.join(ROOT_DIR, "Metadata/column_name.json")
INPUT_METADATA_PATH = os.path.join(ROOT_DIR, "Metadata/feature_name.json")

def load_models(vae_path: str, iso_path: str, input_dim: int, z_dim: int) -> tuple:
    '''Load VAE and ISO models from local paths'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    vae_model = MyVAE(input_dim, z_dim).to(device)
    vae_model.load_state_dict(torch.load(vae_path, map_location=device))
    vae_model.eval()

    iso_model = joblib.load(iso_path)

    return vae_model, iso_model, device


def load_pipeline(vae_pipeline_path: str, iso_pipeline_path: str) -> tuple:
    '''Load VAE and ISO transform rules from local paths'''
    vae_pipeline = joblib.load(vae_pipeline_path)

    iso_pipeline = joblib.load(iso_pipeline_path)

    return vae_pipeline, iso_pipeline

def load_columns_info(col_metadata_path: str, input_metadata_path: str)-> tuple:
    '''Load the column info'''
    with open(col_metadata_path, "r") as f:
        columns = json.load(f)
    with open(input_metadata_path, "r") as f:
        temp_dic = json.load(f)
    input_dim = temp_dic["No. of rows"][0]

    return columns, input_dim

def align_columns(df: pd.DataFrame, column_list: list) -> pd.DataFrame:
    '''Align dataframe columns to match training columns'''
    for col in column_list:
        if col not in df.columns:
            df[col] = np.nan
        
    df = df[column_list]
    return df



def run_pipeline(df_t, df_i, VAE_model, ISO_model, VAE_pipeline, ISO_pipeline, columns, device):
    '''Full pipeline from data to prediction'''
    
    vae_w = 0.9
    iso_w = 0.1
    threshold = 0.15
    batch_size = 2048

    transaction_list = list(columns["transaction_list"])
    identity_list = list(columns["Identity_list"])

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
    

    # Split for VAE and ISO
    v_cols = [c for c in df.columns if c.startswith("V")]
    df_vae = df.drop(columns=v_cols)
    df_iso = df.copy()

    # Apply transform rules
    df_vae_transformed = VAE_pipeline.transform(df_vae).astype("float32")
    df_iso_transformed = ISO_pipeline.transform(df_iso).astype("float32")

    # Get scores
    tensor = torch.tensor(df_vae_transformed, dtype=torch.float32)
    test_dataloader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=False)
    vae_score = evaluate_vae(model=VAE_model, test_dataloader=test_dataloader, device=device)
    iso_score = evaluate_iso(model=ISO_model, test_X=df_iso_transformed)

    # Ensemble
    ensemble_scores = vae_w * vae_score + iso_w * iso_score
    predictions = (ensemble_scores >= threshold).astype(int)

    # Save results
    results = pd.DataFrame({
        "ensemble_score": ensemble_scores,
        "prediction": predictions
    })

    return results