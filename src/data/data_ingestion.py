## Importing necessary libraries:
import os

## Data manipulation
import numpy as np
import pandas as pd

## Logging
import logging

## yaml
import yaml

## absolute path for data
from config import RAW_TRANSACTION_PATH, RAW_IDENTITY_PATH

## file saving
import json

## Setting seed 
np.random.seed(26)

from utils import load_params, setup_logger

logger = setup_logger("data_ingestion","ingestion_error.log")

def load_transaction_data(raw_transaction_data_path: str, no_rows: int)-> pd.DataFrame:
    '''loads transaction dataset with columns that have more than 70% valid entries'''
    try:
        # get total number of rows
        total_rows = sum(1 for _ in open(raw_transaction_data_path)) - 1
        
        # generate a random list of row number
        keep_idx = np.random.choice(total_rows, size=no_rows, replace=False)
        keep_idx = set(keep_idx)
        
        # read the sample_df
        sample_transaction_df = pd.read_csv(raw_transaction_data_path,skiprows=lambda x: x != 0 and (x-1) not in keep_idx)
        logger.debug("Sample transaction data loaded from: %s",raw_transaction_data_path)
        
        # get a list of columns to keep
        cols_to_keep = [c for c in sample_transaction_df.columns if sample_transaction_df[c].isna().mean() < 0.7]
        
        # load all the rows with relevant columns
        df_transaction = pd.read_csv(raw_transaction_data_path,usecols=cols_to_keep)
        logger.debug("Transaction data loaded from: %s",raw_transaction_data_path)
        return df_transaction
    except pd.errors.ParserError as e:
        logger.error("Failed to parse CSV file: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error encountered while loading sample data: %s", e)
        raise

def load_identity_Data(raw_identity_data_path:str)-> pd.DataFrame:
    '''loads identity dataset'''
    try:
        df_identity = pd.read_csv(raw_identity_data_path)
        logger.debug("Identity data loaded from: %s",raw_identity_data_path)
        return df_identity
    except pd.errors.ParserError as e:
        logger.error("Failed to parse CSV file: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error encountered while loading sample data: %s", e)
        raise

def clean_V_cols(df: pd.DataFrame)-> pd.DataFrame:
    '''Preprocess the V-columns'''
    try:
        # V_columns processing
        v_cols = [c for c in df.columns if c.startswith('V')]
        v_cols_sorted = df[v_cols].isna().mean().sort_values(ascending= True).index.tolist()
        # correlation matrix
        corr_matrix = df[v_cols_sorted].corr().abs() 
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [c for c in upper.columns if any(upper[c] > 0.75)]
        # Sparse columns
        dominant_proportion = df[v_cols].apply(lambda x: x.value_counts(normalize=True).iloc[0])
        sparse_v_cols = dominant_proportion[dominant_proportion > 0.90].index.tolist()
        to_drop.extend(sparse_v_cols)
        to_drop = list(set(to_drop))
        # drop columns
        df = df.drop(columns=to_drop)
        logger.debug("V-columns cleaned, Number of V-columns left: %d", len([c for c in df.columns if c.startswith('V')]))
        return df
    except Exception as e:
        logger.error("V-columns cleaning failed: %s", e)
        raise

def clean_C_cols(df: pd.DataFrame)-> pd.DataFrame:
    '''Preprocess the C-columns'''
    try:
        # C_columns processing
        C_cols = [c for c in df.columns if c.startswith('C')]
        # order according to sparsity
        dominant_proportion = df[C_cols].apply(lambda x: x.value_counts(normalize=True).iloc[0])
        sparse_c_cols = dominant_proportion[dominant_proportion > 0.90].index.tolist()
        C_cols_sorted = dominant_proportion.sort_values(ascending=True).index.tolist()
        # correlation matrix
        corr_matrix = df[C_cols_sorted].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]
        to_drop.extend(sparse_c_cols)
        to_drop = list(set(to_drop))
        # drop columns
        df = df.drop(columns=to_drop)
        logger.debug("C-columns cleaned, Number of C-columns left: %d", len([c for c in df.columns if c.startswith('c')]))
        return df
    except Exception as e:
        logger.error("C-columns cleaning failed: %s", e)
        raise

def clean_D_cols(df: pd.DataFrame)-> pd.DataFrame:
    '''Preprocess the D-columns'''
    try:
        # D_columns processing
        D_cols = [c for c in df.columns if c.startswith('D')]
        D_cols_sorted = df[D_cols].isna().mean().sort_values(ascending= True).index.tolist()
        # correlation matrix
        corr_matrix = df[D_cols_sorted].corr().abs() 
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [c for c in upper.columns if any(upper[c] > 0.85)]
        # drop columns
        df = df.drop(columns=to_drop)
        logger.debug("D-columns cleaned, Number of D-columns left: %d", len([c for c in df.columns if c.startswith('D')]))
        return df
    except Exception as e:
        logger.error("D-columns cleaning failed: %s", e)
        raise

def clean_card_cols(df: pd.DataFrame)-> pd.DataFrame:
    '''Preprocess the card-columns'''
    try:
        # card_columns processing
        card_cols = [c for c in df.columns if c.startswith('card')]
        to_drop = []
        for cols in card_cols:
            non_null = df[cols].dropna()
            dominant = non_null.value_counts(normalize=True).iloc[0]
            if dominant> 0.85:
                to_drop.append(cols)
        df = df.drop(columns=to_drop)
        logger.debug("card-columns cleaned, Number of card-columns left: %d", len([c for c in df.columns if c.startswith('card')]))
        return df
    except Exception as e:
        logger.error("card-columns cleaning failed: %s", e)
        raise

def clean_addr_cols(df: pd.DataFrame)-> pd.DataFrame:
    '''Preprocess the addr-columns'''
    try:
        # addr_columns processing
        addr_cols = [c for c in df.columns if c.startswith('addr')]
        to_drop = []
        for cols in addr_cols:
            non_null = df[cols].dropna()
            dominant = non_null.value_counts(normalize=True).iloc[0]
            if dominant> 0.85:
                to_drop.append(cols)
        df = df.drop(columns=to_drop)
        logger.debug("addr-columns cleaned, Number of addr-columns left: %d", len([c for c in df.columns if c.startswith('addr')]))
        return df
    except Exception as e:
        logger.error("addr-columns cleaning failed: %s", e)
        raise

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

def clean_pemail_col(df: pd.DataFrame)-> pd.DataFrame:
    '''Bin P_emaildomain into 6 categories based on provider'''
    try:
        df["P_emaildomain"] = df["P_emaildomain"].apply(clean_email)
        logger.debug("email-columns binned successfully")
        return df
    except Exception as e:
        logger.error("P_emaildomain-column binning failed: %s", e)
        raise

def clean_TranAmt_col(df: pd.DataFrame)-> pd.DataFrame:
    '''Remove outliers in TransactionAmt column'''
    try:
        # Getting the 99th percentile value
        cap_value = df["TransactionAmt"].quantile(0.99)
        while df["TransactionAmt"].max() > cap_value*10:
            idmax = df[df['TransactionAmt'] == max(df["TransactionAmt"])].index
            df = df.drop(index=idmax[0])
        logger.debug("TransactionAmt-column cleaned")
        return df
    except Exception as e:
        logger.error("TransactionAmt-column outlier removal failed: %s", e)
        raise

def extract_temporal_features(df: pd.DataFrame)-> pd.DataFrame:
    '''Extract hours and days of the week information from datetime column'''
    try:
        # hours information
        df['hour'] = (df['TransactionDT'] // 3600) % 24
        # days of the week information
        df['day_of_week'] = (df['TransactionDT'] // (3600 * 24)) % 7
        logger.debug("Hours and days of the week information successfully extracted")
        return df
    except Exception as e:
        logger.error("Hours and days of the week extraction failed: %s", e)
        raise

def clean_transaction_table(df: pd.DataFrame)-> pd.DataFrame:
    try:
        df = clean_V_cols(df)
        df = clean_C_cols(df)
        df = clean_D_cols(df)
        df = clean_card_cols(df)
        df = clean_addr_cols(df)
        df = clean_pemail_col(df)
        df = clean_TranAmt_col(df)
        df = extract_temporal_features(df)
        logger.debug("Transaction table cleaned successfully, shape: %s", df.shape)
        return df
    except Exception as e:
        logger.error("Transaction table cleaning failed: %s", e)
        raise

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

def clean_id_col(df: pd.DataFrame)-> pd.DataFrame:
    '''Preprocessing the id columns'''
    try:
        # make a list of id columns
        id_col = [c for c in df.columns if c.startswith("id")]
        # drop nearly empty columns
        to_drop = [c for c in id_col if df[c].isna().mean() > 0.9]
        # make seperate list of numerical and categorical columns
        id_num_col = df[id_col].select_dtypes(include = "number").columns.tolist()
        id_cat_col = df[id_col].select_dtypes(include = "object").columns.tolist()
        # remove nearly sparse columns
        dominant_proportion = df[id_num_col].apply(lambda x: x.value_counts(normalize=True).iloc[0])
        sparse_id_num_cols = dominant_proportion[dominant_proportion > 0.90].index.tolist()
        to_drop.extend(sparse_id_num_cols)
        check_df = df[id_num_col].fillna(df[id_num_col].median()) # this is temporary filling to check the sparsness.
        for col in id_num_col:
            dominant = check_df[col].value_counts(normalize=True).iloc[0:2]
            if dominant.iloc[0] > 0.50 and dominant.sum()>0.80:
                to_drop.append(col)
        to_drop = list(set(to_drop))
        df = df.drop(columns=to_drop)
        
        # cleaning categorical cols
        df["id_30"] = df["id_30"].apply(clean_id30)
        df["id_31"] = df["id_31"].apply(clean_id31)
        df["id_33"] = df["id_33"].apply(bin_resolution)
        logger.debug("id-columns cleaned, Number of id-columns left: %d", len([c for c in df.columns if c.startswith('id')]))
        return df
    except Exception as e:
        logger.error("id-columns cleaning failed: %s", e)
        raise

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

def bin_deviceinfo_col(df: pd.DataFrame)-> pd.DataFrame:
    '''Binning device info column'''
    try:
        df["DeviceInfo"] = df["DeviceInfo"].apply(clean_device_info)
        logger.debug("Device info binned successfully")
        return df
    except Exception as e:
        logger.error("DeviceInfo binning cleaned: %s", e)
        raise

def clean_identity_table(df: pd.DataFrame)->pd.DataFrame:
    try:
        # identity cleaning
        df = clean_id_col(df)
        df = bin_deviceinfo_col(df)
        
        logger.debug("Identity table cleaned successfully, shape: %s", df.shape)
        return df
    except Exception as e:
        logger.error("Identity table cleaning failed: %s", e)
        raise

def merge_df(df_t: pd.DataFrame, df_i: pd.DataFrame)-> pd.DataFrame:
    '''Merging the transaction and identity table using TransactionID'''
    try:
        # left join on the transaction table using TransactionID
        df = df_t.merge(df_i, on="TransactionID", how="left")
        df = df.drop(columns=["TransactionID"])
        logger.debug("Tables merged successfully")
        return df
    except Exception as e:
        logger.error("Merge unsuccessfull: %s", e)
        raise

def apply_log_transforms(df: pd.DataFrame)-> tuple:
    '''Apply log+1 transform on skewed columns'''
    try:
        # get the numerical columns
        num_col = df.select_dtypes(include="number").columns.tolist()
        if "isFraud" in num_col:
            num_col.remove("isFraud")
        # skewed columns with minimum > 0
        skewness = df[num_col].skew().sort_values(ascending=False)
        to_log = skewness[skewness > 1.0].index.tolist()
        to_log = [col for col in to_log if df[col].min() >=0]
        # perform log transform
        for col in to_log:
            df[col] = np.log1p(df[col])
        logger.debug("log-transform done successfully")
        return df,to_log
    except Exception as e:
        logger.error("Merge unsuccessfull: %s", e)
        raise

def save_dataset(df: pd.DataFrame, data_path: str) -> None:
    '''Save the cleaned dataset to the processed data directory'''
    try:
        raw_data_path = os.path.join(data_path,"raw")
        # Create the relevant directory if it doesnt exist
        os.makedirs(raw_data_path, exist_ok=True)
        # save the dataframe
        df.to_csv(os.path.join(raw_data_path,"full_dataset.csv"), index=False)
        logger.debug("Dataset saved successfully to %s, shape: %s", raw_data_path, df.shape)
    except Exception as e:
        logger.error("Dataset saving failed: %s", e)
        raise

def save_column_name(t_list: list,i_list, log_list: list, file_path: str):
    '''Save columns name for serving time'''
    try:
        feature_info = {"transaction_list": t_list,"Identity_list": i_list, "log_list": log_list}

        with open(file_path, "w") as f:
            json.dump(feature_info,f)
        logger.debug("feature info saved: %s", file_path)

    except Exception as e:
        logger.error("feature info failed to be saved: %s", e)
        raise

def main():
    try:
        # Load params form YAML file
        params = load_params(params_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"../params.yaml"))
        no_rows = params["data_ingestion"]["no_rows"]
        
        # Load the transaction dataset
        df_t = load_transaction_data(RAW_TRANSACTION_PATH, no_rows = no_rows)
        df_t = clean_transaction_table(df_t)
        t_list = [c for c in df_t.columns if c not in ["isFraud", "hour", "day_of_week"]]
        # Load the identity dataset
        df_i = load_identity_Data(RAW_IDENTITY_PATH)
        df_i = clean_identity_table(df_i)
        i_list = df_i.columns.to_list()
        # Merge the table
        df = merge_df(df_t, df_i)
        # Log transform
        df,to_log = apply_log_transforms(df)

        save_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"../Metadata")
        os.makedirs(save_file_path, exist_ok=True)
        
        save_column_name(t_list=t_list,i_list=i_list, log_list = to_log, file_path= os.path.join(save_file_path,"column_name.json"))

        
        # save dataset
        save_dataset(df, data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"../processedData"))
        
        logger.debug("Data ingestion completed successfully")
    except Exception as e:
        logger.error("Data ingestion pipeline failed: %s", e)
        raise

if __name__== "__main__":
    main()