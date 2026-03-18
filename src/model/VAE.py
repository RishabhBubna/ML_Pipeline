## Data manipulation
import numpy as np

## Deep learning
import torch
from torch.utils.data import Dataset
from torch import from_numpy
from torch import nn

## Dataset class

class Frauddataset(Dataset):
    '''Loads preprocessed numpy arrays into PyTorch.
    Uses from_numpy to avoid copying data in memory.'''
    def __init__(self, featuresfile, labelfile = None):
        features =  np.load(featuresfile)
        self.x = from_numpy(features)
        del features
        
        if labelfile is not None:
            label = np.load(labelfile)
            self.y = from_numpy(label).float()
            del label
            
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self,idx):
        
        features = self.x[idx]
        
        if hasattr(self, 'y') and self.y is not None:
            return features, self.y[idx]
    
        return features
            
## VAE class function

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
    
def vae_loss_function(reconstructed_x: torch.Tensor, 
                      x: torch.Tensor, 
                      mu: torch.Tensor, 
                      logvar: torch.Tensor, 
                      beta: float) -> torch.Tensor:
    '''VAE loss = MSE reconstruction loss + beta * KLD loss'''
    
    recon_loss = nn.functional.mse_loss(reconstructed_x, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta*kld_loss

