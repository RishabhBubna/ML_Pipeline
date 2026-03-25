## Importing necessary libraries:
import os

## Data manipulation
import numpy as np

## Plotting 
import matplotlib.pyplot as plt

## Setting the plot style
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 11

## Machine learning
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

## Deep learning
import torch
from torch.utils.data import Dataset, DataLoader
from torch import from_numpy
from torch import nn

## Experiment tracking
import mlflow
import mlflow.sklearn
import mlflow.pytorch
from mlflow.models import infer_signature
import boto3
import awscli

## File saving
import joblib
import json

from model.VAE import MyVAE, Frauddataset

from utils import load_params, setup_logger

logger = setup_logger("model_evaluation", "evaluation_error.log")

def load_data_vae(file_path_feature: str,file_path_label:str, batch_size: int, shuffle:bool)-> DataLoader:
    '''Load numpy matrix into the dataloader to hand it to the VAE'''
    try:
        dataset = Frauddataset(featuresfile = file_path_feature, labelfile = file_path_label)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        logger.debug("data loaded into the dataloader for vae with shape: %s", (len(dataset),{dataset.x.shape[1]}))
        return dataloader
    except Exception as e:
        logger.error("dataloading failed: %s", e)
        raise

def load_data_iso(file_path_feature: str,file_path_label:str)-> tuple:
    '''Load data for evaluation of the Isolation forest'''
    try:

        test_X = np.load(file_path_feature)
        test_y = np.load(file_path_label)

        logger.debug("Data loaded for isolation forest, shape of data: %s", test_X.shape)
        logger.debug("Label loaded for isolation forest, shape of data: %s", test_y.shape)
        return test_X, test_y
    except Exception as e:
        logger.error("test data could not be loaded for isolation forest: %s", e)
        raise

def load_vae(vae_path: str, model):
    '''Load the VAE model'''
    try:
        
        model.load_state_dict(torch.load(vae_path))
        logger.debug("VAE model loaded")
        return model
    except Exception as e:
        logger.error("Failed to load VAE model: %s", e)
        raise

def load_iso(iso_path: str):
    '''Load the isolation forest model'''
    try:

        iso_forest = joblib.load(iso_path)
        logger.debug("Isolation forest model loaded.")
        return iso_forest
    except Exception as e:
        logger.error("Failed to load Isolation forest model: %s", e)
        raise

def evaluate_vae(model, test_dataloader: DataLoader, device):
    '''Evaluate the score for the vae model'''
    try:
        model.eval()

        reconstruction_errors = []

        with torch.no_grad():
            for X, y in test_dataloader:
                X = X.to(device)
                reconstructed, _, _ = model(X)
                errors = torch.mean((X - reconstructed) ** 2, dim=1)
        
                errors_np = errors.cpu().numpy()
        
                reconstruction_errors.extend(errors_np)

        reconstruction_errors = np.array(reconstruction_errors)
        vae_scores = reconstruction_errors.reshape(-1, 1)
        vae_normalized = MinMaxScaler().fit_transform(vae_scores).flatten()
        
        logger.debug("VAE_score calculated")
        return vae_normalized
    except Exception as e:
        logger.error("Evaluation of VAE failed: %s", e)
        raise

def evaluate_iso(model, test_X: np.ndarray):
    '''Evaluate the score for the Isolation forest model'''
    try:

        iso_scores = -model.decision_function(test_X)
        iso_scores_reshaped = iso_scores.reshape(-1, 1)
        iso_normalized = MinMaxScaler().fit_transform(iso_scores_reshaped).flatten()

        logger.debug("iso_score calculated")
        return iso_normalized
    except Exception as e:
        logger.error("Evaluation of isloation forest failed: %s", e)
        raise

def log_metrics(vae_w: float, iso_w: float, vae_normalized, iso_normalized, test_y: np.ndarray):
    '''log metrics and PR-curve to MLflow'''
    try:

        ensemble_scores = vae_w * vae_normalized + iso_w * iso_normalized
        auroc = roc_auc_score(test_y, ensemble_scores)
        ap = average_precision_score(test_y, ensemble_scores)

        mlflow.log_metric("AUROC",auroc)
        mlflow.log_metric("AP", ap)
        
        logger.debug("AUROC and AP logged: %s", (auroc, ap))
        ens_precision, ens_recall, _ = precision_recall_curve(test_y, ensemble_scores)
        random_baseline = test_y.mean()

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(ens_recall, ens_precision, color="darkorange", linewidth=2,label=f"Ensemble 0.9/0.1 (AP = {average_precision_score(test_y, ensemble_scores):.4f})")
        ax.axhline(y=random_baseline, color="crimson", linestyle="--",label=f"Random baseline ({random_baseline:.3f})")

        ax.set_title("Precision-Recall Curve")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend()

        plt.tight_layout()


        save_PR_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"../Metrics/Plots")
        os.makedirs(save_PR_path, exist_ok=True)

        plt.savefig(os.path.join(save_PR_path, "PR_curve.png"))

        mlflow.log_artifact(os.path.join(save_PR_path, "PR_curve.png"))
        plt.close()
        logger.debug("PR-curve logged.")
    except Exception as e:
        logger.error("Logging to MLflow failed: %s", e)
        raise

def save_ensemble_info(run_id: str, vae_path: str, iso_path: str, file_path: str)-> None:
    '''Save run_id and model path to a JSON file'''
    try:

        model_info = {"run_id": run_id, "VAE_path": vae_path, "ISO_path": iso_path}

        with open(file_path, 'w') as f:
            json.dump(model_info, f, indent=4)
        logger.debug("model info saved: %s", file_path)

    except Exception as e:
        logger.error("model info failed to be saved: %s", e)
        raise

def main():
    mlflow.set_tracking_uri("http://ec2-63-182-216-170.eu-central-1.compute.amazonaws.com:5000")

    mlflow.set_experiment("DVC_Ensemble_run")

    with mlflow.start_run() as run:
        try:

            params = load_params(params_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"../params.yaml"))

            vae_w = params["model_evaluation"]["vae_w"]
            iso_w = params["model_evaluation"]["iso_w"]
            seed = params["train_vae"]["seed"]
            batch_size = params["train_vae"]["batch_size"]
            z_dim = params["train_vae"]["z_dim"]

            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)

            for section, section_params in params.items():
                for key, value in section_params.items():
                    mlflow.log_param(f"{section}.{key}", value)

            logger.debug("parameters logged to MLflow")

            root_dic = os.path.join(os.path.dirname(os.path.abspath(__file__)),"../")
            file_path_feature_vae = os.path.join(root_dic, "processedData/preprocessed/testset_VAE.npy")
            file_path_label = os.path.join(root_dic, "processedData/preprocessed/testset_label.npy")
            test_dataloader =  load_data_vae(file_path_feature = file_path_feature_vae,file_path_label= file_path_label, batch_size=batch_size, shuffle=False)

            file_path_feature_iso = os.path.join(root_dic, "processedData/preprocessed/testset_Iso.npy")
            test_X, test_y = load_data_iso(file_path_feature =  file_path_feature_iso,file_path_label= file_path_label)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            input_dim = next(iter(test_dataloader))[0].shape[1]
            model = MyVAE(input_dim,z_dim).to(device)

            vae_path = os.path.join(root_dic,"model/best_vae.pt")
            vae_model = load_vae(vae_path= vae_path, model = model)

            logger.debug("VAE loaded")

            iso_path = os.path.join(root_dic,"model/iso_forest.pkl")
            iso_model = load_iso(iso_path = iso_path)

            logger.debug("Isolation forest loaded")

            vae_score = evaluate_vae(model = vae_model, test_dataloader= test_dataloader, device = device)
            iso_score = evaluate_iso(model = iso_model, test_X= test_X)

            log_metrics(vae_w = vae_w, iso_w = iso_w, vae_normalized = vae_score, iso_normalized = iso_score, test_y = test_y)

            logger.debug("metrics logged")

            signature_iso = infer_signature(test_X[:5], iso_score[:5])
            
            example_input = next(iter(test_dataloader))[0][:5].to(device)
            example_output, _, _ = model(example_input)
            signature_vae = infer_signature(example_input.cpu().numpy(), example_output.detach().cpu().numpy())

            mlflow.pytorch.log_model(pytorch_model=vae_model,artifact_path="VAE_model",signature=signature_vae,input_example=example_input.cpu().numpy())
            artifact_uri = mlflow.get_artifact_uri()
            vae_artifact_path = os.path.join(artifact_uri,"VAE_model")

            mlflow.sklearn.log_model(sk_model=iso_model,artifact_path="iso_forest_model",signature=signature_iso,input_example = test_X[:5])
            artifact_uri = mlflow.get_artifact_uri()
            iso_artifact_path = os.path.join(artifact_uri,"iso_forest_model")

            logger.debug("models logged")

            save_ensemble_info(run_id= run.info.run_id, vae_path = vae_artifact_path, iso_path= iso_artifact_path, file_path= os.path.join(root_dic,"Metadata/experiment_info.json"))

            mlflow.set_tag("model_type", "VAE+ISO")
            mlflow.set_tag("Task", "Anomaly detection")
            mlflow.set_tag("dataset","IEEE-CIS")

        except Exception as e:
            logger.error("Model evaluation failed")
            raise

if __name__ == "__main__":
    main()





