import os

## Data manipulation
import numpy as np
import pandas as pd

## deep learning
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn

## File saving
import joblib
import json

# from data.data_ingestion import (extract_temporal_features,clean_id30,clean_id31,bin_resolution,clean_device_info,merge_df)
# from model.model_evaluation import (evaluate_vae,evaluate_iso)
# from model.VAE import MyVAE


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

VAE_MODEL_PATH = os.path.join(ROOT_DIR, "model/best_vae.pt")
ISO_MODEL_PATH = os.path.join(ROOT_DIR, "model/iso_forest.pkl")
VAE_PIPELINE_PATH = os.path.join(ROOT_DIR, "model/pipeline/transform_rule_VAE.pkl")
ISO_PIPELINE_PATH = os.path.join(ROOT_DIR, "model/pipeline/transform_rule_Iso.pkl")
VAE_SCALER_PATH = os.path.join(ROOT_DIR, "model/pipeline/vae_scaler.pkl")
ISO_SCALER_PATH = os.path.join(ROOT_DIR, "model/pipeline/iso_scaler.pkl")
COLUMN_METADATA_PATH = os.path.join(ROOT_DIR, "Metadata/column_name.json")
INPUT_METADATA_PATH = os.path.join(ROOT_DIR, "Metadata/feature_name.json")

def clean_email(x):
        x = str(x).lower()
        if 'gmail' in x: return 'gmail'
        if 'yahoo' in x or "ymail" in x: return 'yahoo'
        if 'hotmail' in x: return 'hotmail'
        if 'icloud' in x or "mac" in x: return 'icloud'
        if 'anonymous' in x: return 'anonymous'
        if 'outlook' in x or 'live' in x or 'msn' in x: return 'microsoft'
        if 'missing' in x: return 'missing'
        if 'nan' in x: return np.nan
        return 'other'

def extract_temporal_features(df: pd.DataFrame)-> pd.DataFrame:
    '''Extract hours and days of the week information from datetime column'''
    # hours information
    df['hour'] = (df['TransactionDT'] // 3600) % 24
    # days of the week information
    df['day_of_week'] = (df['TransactionDT'] // (3600 * 24)) % 7
    return df
    

def clean_id30(x):
    x = str(x).lower()
    if 'missing' in x: return 'missing'
    if 'windows' in x: return 'windows'
    if 'ios' in x or 'iphone' in x: return 'ios'
    if 'mac' in x: return 'mac'
    if 'android' in x: return 'android'
    if 'linux' in x: return 'linux'
    if 'nan' in x: return np.nan
    return 'other'

def clean_id31(x):
    x = str(x).lower()
    if 'missing' in x: return 'missing'
    if 'chrome' in x: return 'chrome'
    if 'safari' in x: return 'safari'
    if 'ie' in x: return 'ie'
    if 'edge' in x: return 'edge'
    if 'firefox' in x: return 'firefox'
    if 'samsung' in x: return 'samsung_browser'
    if 'nan' in x: return np.nan
    return 'other'

def bin_resolution(x):
    x = str(x).lower()
    if 'nan' in x: return np.nan
    if 'missing' in x: return 'missing'
    try:
        width, height = map(int, x.split('x'))
    except:
        return 'other'
    # Mobile resolutions
    if (width <= 1334 and height <= 750) or (width <= 750 and height <= 1334):
        return 'mobile'
    # HD
    if width <= 1366 and height <= 768:
        return 'hd'
    # FHD
    if width <= 1920 and height <= 1200:
        return 'fhd'
    # 2K
    if width <= 2880 and height <= 1800:
        return '2k'
    # Tablet 
    if (width == 2048 and height == 1536) or (width == 2732 and height == 2048) or (width == 2224 and height == 1668):
        return 'tablet'
    # 4K
    if width >= 3840:
        return '4k'
    return 'other'

def clean_device_info(x):
    x = str(x).lower()
    if 'missing' in x: return 'missing'
    if 'windows' in x: return 'windows'
    if 'ios' in x or 'iphone' in x: return 'ios'
    if 'mac' in x: return 'mac'
    if 'trident' in x: return 'ie'
    if 'rv:' in x: return 'firefox'
    if 'huawei' in x or 'ale-' in x or 'cam-' in x or 'hi6210' in x: return 'huawei'
    if 'samsung' in x or 'sm-' in x: return 'samsung'
    if 'lg-' in x: return 'lg'
    if 'moto' in x: return 'motorola'
    if 'android' in x: return 'android'
    if 'nan' in x: return np.nan
    return 'other'

def merge_df(df_t: pd.DataFrame, df_i: pd.DataFrame)-> pd.DataFrame:
    '''Merging the transaction and identity table using TransactionID'''
    # left join on the transaction table using TransactionID
    df = df_t.merge(df_i, on="TransactionID", how="left")
    df = df.drop(columns=["TransactionID"])
    return df

class MyVAE(nn.Module):
    '''Variational Autoencoder for anomaly detection.
    Trained on normal transactions only.'''
    def __init__(self,input_dim, z_dim):
        super().__init__()
        
        ## Encoding architecture
        
        self.encode_arc = nn.Sequential(
                            nn.Linear(input_dim,64),
                            nn.LeakyReLU(),
                            nn.Linear(64,32),
                            nn.LeakyReLU(),
                            nn.Linear(32,16),
                            nn.LeakyReLU(),
        )
        
        self.mu_head = nn.Linear(16, z_dim)
        self.logvar_head = nn.Linear(16, z_dim)
        
        ## Decoding architecture
        
        self.decode_arc = nn.Sequential(
                            nn.Linear(z_dim, 16),
                            nn.ReLU(),
                            nn.Linear(16,32),
                            nn.ReLU(),
                            nn.Linear(32,64),
                            nn.ReLU(),
                            nn.Linear(64, input_dim)
        )
        
        
        
        
    def encode(self,x):
        h = self.encode_arc(x)
        return self.mu_head(h), self.logvar_head(h)
    
    def reparameterize(self, mu, logvar):
        sigma = torch.exp(0.5*logvar)
        epsilon = torch.randn_like(sigma)
        reparam = mu + epsilon*sigma
        return reparam
    
    def decode(self,z):
        h = self.decode_arc(z)
        return h
    
    def forward(self,x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu,logvar)
        reconstructed_x = self.decode(z)
        return reconstructed_x, mu, logvar

def evaluate_vae(model, test_dataloader: DataLoader, device, scaler):
    '''Evaluate the score for the vae model'''
    model.eval()

    reconstruction_errors = []

    with torch.no_grad():
        for batch in test_dataloader:
            X = batch[0].to(device)
            reconstructed, _, _ = model(X)
            errors = torch.mean((X - reconstructed) ** 2, dim=1)
    
            errors_np = errors.cpu().numpy()
    
            reconstruction_errors.extend(errors_np)

    reconstruction_errors = np.array(reconstruction_errors).reshape(-1, 1)
    vae_normalized = scaler.transform(reconstruction_errors).flatten()
    return vae_normalized


def evaluate_iso(model, test_X: np.ndarray, scaler):
    '''Evaluate the score for the Isolation forest model'''

    iso_scores = -model.decision_function(test_X)
    iso_normalized = scaler.transform(iso_scores.reshape(-1, 1)).flatten()

    return iso_normalized


def load_models(vae_path: str, iso_path: str, input_dim: int, z_dim: int) -> tuple:
    '''Load VAE and ISO models from local paths'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    loaded = torch.load(vae_path, map_location=device, weights_only=False)
    
    if isinstance(loaded, dict):
        vae_model = MyVAE(input_dim, z_dim).to(device)
        vae_model.load_state_dict(loaded)
    else:
        vae_model = loaded.to(device)
    
    vae_model.eval()
    iso_model = joblib.load(iso_path)

    return vae_model, iso_model, device


def load_pipeline(vae_pipeline_path: str, iso_pipeline_path: str) -> tuple:
    '''Load VAE and ISO transform rules from local paths'''
    vae_pipeline = joblib.load(vae_pipeline_path)

    iso_pipeline = joblib.load(iso_pipeline_path)

    return vae_pipeline, iso_pipeline

def load_scaler(vae_scaler_path: str, iso_scaler_path: str) -> tuple:
    '''Load VAE and ISO transform rules from local paths'''
    vae_scaler = joblib.load(vae_scaler_path)

    iso_scaler = joblib.load(iso_scaler_path)

    return vae_scaler, iso_scaler

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



def run_pipeline(df_t, df_i, VAE_model, ISO_model, VAE_pipeline, ISO_pipeline,VAE_scaler, ISO_scaler, columns, device):
    '''Full pipeline from data to prediction'''
    
    vae_w = 0.9
    iso_w = 0.1
    threshold = 0.0779
    batch_size = 2048

    transaction_list = list(columns["transaction_list"])
    identity_list = list(columns["Identity_list"])

    # Clean
    df_t["P_emaildomain"] = df_t["P_emaildomain"].apply(clean_email)
    df_t = extract_temporal_features(df_t)
    transaction_list.extend(["hour","day_of_week"])
    # Align columns
    df_t = align_columns(df_t, transaction_list)

    if not df_i.empty:
        df_i["id_30"] = df_i["id_30"].apply(clean_id30)
        df_i["id_31"] = df_i["id_31"].apply(clean_id31)
        df_i["id_33"] = df_i["id_33"].apply(bin_resolution)
        df_i["DeviceInfo"] = df_i["DeviceInfo"].apply(clean_device_info)
        df_i = align_columns(df_i, identity_list)
    else:
        df_i = pd.DataFrame(columns=["TransactionID"])
        df_i = align_columns(df_i, identity_list)
    # Merge
    df = merge_df(df_t, df_i)

    # Log transform
    for col in columns["log_list"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
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
    vae_score = evaluate_vae(model=VAE_model, test_dataloader=test_dataloader, device=device, scaler= VAE_scaler)
    iso_score = evaluate_iso(model=ISO_model, test_X=df_iso_transformed, scaler= ISO_scaler)

    # Ensemble
    ensemble_scores = vae_w * vae_score + iso_w * iso_score
    predictions = (ensemble_scores >= threshold).astype(int)

    # Save results
    results = pd.DataFrame({
        "ensemble_score": ensemble_scores,
        "prediction": predictions
    })

    return results