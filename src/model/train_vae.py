## Importing necessary libraries:
import os

## Data manipulation
import numpy as np
import pandas as pd

## Metric
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

## Deep learning
import torch
from torch.utils.data import Dataset, DataLoader
from torch import from_numpy
from torch import nn

from model.VAE import MyVAE, Frauddataset, vae_loss_function

from utils import load_params, setup_logger

logger = setup_logger("training_VAE", "VAE_error.log")



def load_data_dataloader(file_path_feature: str,file_path_label:str, batch_size: int, shuffle:bool)-> DataLoader:
    '''Load numpy matrix into the dataloader to hand it to the model'''
    try:
        dataset = Frauddataset(featuresfile = file_path_feature, labelfile = file_path_label)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        logger.debug("data loaded into the dataloader with shape: %s", (len(dataset),{dataset.x.shape[1]}))
        return dataloader
    except Exception as e:
        logger.error("dataloading failed: %s", e)
        raise

def train_1epoch(model: MyVAE, train_dataloader: DataLoader, optimizer: torch.optim.Optimizer, beta: float, device: torch.device)-> float:
    '''Training one epoch'''
    try:
        model.train()
    
        train_loss = 0
        size = len(train_dataloader.dataset)
    
        for batch, x in enumerate(train_dataloader):
            x = x.to(device)
            optimizer.zero_grad()
            reconstructed_x, mu, logvar = model(x)
            loss = vae_loss_function(reconstructed_x, x, mu, logvar, beta)
        
       
            loss.backward()
            optimizer.step()
        
        train_loss += loss.item()
        Avg_loss = train_loss / size
        return Avg_loss
    except Exception as e:
        logger.error("training failed at one epch level: %s", e)
        raise

def AUROC_AP_1epcoh(model, test_dataloader, device):
    '''Calculate AUROC and AP after one epoch'''
    try:

        model.eval()
        reconstruction_errors = []
        all_labels = []
    
        with torch.no_grad():
            for X, y in test_dataloader:
                X = X.to(device)
                reconstructed, mu, logvar = model(X)
                errors = torch.mean((X - reconstructed) ** 2, dim=1)
                reconstruction_errors.extend(errors.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
            
        reconstruction_errors = np.array(reconstruction_errors)
        all_labels = np.array(all_labels)
    
        auroc = roc_auc_score(all_labels, reconstruction_errors)
        ap = average_precision_score(all_labels, reconstruction_errors)
        return auroc, ap
    except Exception as e:
        logger.error("calculationg of AUROC and AP failed: %s", e)
        raise


def training_loop(params:dict)->None:
    '''The full training loop running over all epochs'''
    try:
        batch_size = params["train_vae"]["batch_size"]
        epochs = params["train_vae"]["epoch"]
        beta = params["train_vae"]["beta"]
        z_dim = params["train_vae"]["z_dim"]
        lr = params["train_vae"]["lr"]
        warmup_epoch = params["train_vae"]["warmup_epochs"]
        
        preprocessed_path =  os.path.join(os.path.dirname(os.path.abspath(__file__)),"../processedData/preprocessed")
        train_path = os.path.join(preprocessed_path, "trainset_VAE.npy")
        test_path = os.path.join(preprocessed_path, "testset_VAE.npy")
        label_path = os.path.join(preprocessed_path, "testset_label.npy")
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"best_vae.pt")

        train_dataloader = load_data_dataloader(file_path_feature = train_path,file_path_label = None, batch_size = batch_size, shuffle=True)
        test_dataloader = load_data_dataloader(file_path_feature = test_path,file_path_label = label_path, batch_size = batch_size, shuffle=False)

        logger.debug("Train and test set loaded into the dataloader.")

        best_ap = 0
        input_dim = next(iter(train_dataloader)).shape[1]

        logger.debug("Input dimension: %s", input_dim)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.debug("Device: %s", device)

        model = MyVAE(input_dim,z_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # list to store results
        # losses = []
        # aurocs = []
        # aps = []
            
        # training
        for epoch in range(epochs):
            avg_loss = train_1epoch(model,train_dataloader,optimizer,beta,device)
            auroc, ap = AUROC_AP_1epcoh(model, test_dataloader, device)
    
            # losses.append(avg_loss)
            # aurocs.append(auroc)
            # aps.append(ap)
    
            if epoch >= warmup_epoch and ap > best_ap:
                best_ap = ap
                torch.save(model.state_dict(), save_path)
                logger.debug("best model saved with (epoch,AP): %s", (epoch,best_ap))

        logger.debug("training complete")
    except Exception as e:
        logger.error("Training to all epochs failed: %s", e)
        raise

def main():
    try:

        params = load_params(params_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"../params.yaml"))
        seed = params["train_vae"]["seed"]
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        training_loop(params)

    except Exception as e:
        logger.error("Training failed: %s", e)
        raise 

if __name__ == "__main__":
    main()