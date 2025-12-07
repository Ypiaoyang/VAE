"""
Image-Guided Graph Variational Autoencoder

This module implements the core CVAE architecture where tissue images serve as
prior conditions rather than reconstruction targets.

Key Components:
- Image Encoder: Processes pre-extracted image features (e.g., from UNI/ResNet)
- Gene Encoder: Processes normalized gene expression
- Prior Network: GAT layers on image features → (μ_p, σ_p)
- Posterior Network: GAT layers on gene+image features → (μ_q, σ_q)
- Decoder: MLP → ZINB parameters (μ, θ, π)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class ImageGuidedGVAE(nn.Module):
    """
    Conditional Graph Variational Autoencoder with Image-Guided Prior
    
    Architecture:
        Encoder (Posterior):  q(z | x, I, A) → N(μ_q, σ²_q)
        Prior (Learned):      p(z | I, A)    → N(μ_p, σ²_p)
        Decoder:              p(x | z)       → ZINB(μ, θ, π)
    
    The key innovation is that the prior network operates ONLY on image and
    spatial graph structure, providing morphology-based guidance independent
    of gene expression.
    
    Args:
        input_dim (int): Number of genes (gene expression dimensionality)
        img_dim (int): Dimensionality of pre-extracted image features
                       (e.g., 1024 for ResNet/UNI embeddings)
        hidden_dim (int): Hidden layer dimensionality
        latent_dim (int): Latent space dimensionality
        num_heads (int): Number of attention heads in GAT layers
        dropout (float): Dropout probability for regularization
    """
    
    def __init__(
        self, 
        input_dim, 
        img_dim, 
        hidden_dim=256, 
        latent_dim=32, 
        num_heads=3,
        dropout=0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.img_dim = img_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        
        # ========================================
        # Feature Encoders
        # ========================================
        
        # Image Feature Encoder
        # Assumes input is already pre-extracted embeddings (e.g., from UNI)
        # Maps image embeddings to hidden space
        self.img_encoder = nn.Sequential(
            nn.Linear(img_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # Gene Expression Encoder
        # Processes normalized gene expression counts
        self.gene_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # ========================================
        # Prior Network p(z | I, A)
        # ========================================
        # CRITICAL: Operates ONLY on image features and graph structure
        # This ensures the prior represents morphological information
        # independent of gene expression
        
        self.p_gat_shared = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=num_heads,
            concat=False,
            dropout=dropout
        )
        
        self.p_mean = nn.Linear(hidden_dim, latent_dim)
        self.p_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # ========================================
        # Posterior Network q(z | X, I, A)
        # ========================================
        # Operates on concatenated gene + image features
        
        self.q_gat_shared = GATConv(
            in_channels=hidden_dim * 2,  # Gene + Image concatenation
            out_channels=hidden_dim,
            heads=num_heads,
            concat=False,
            dropout=dropout
        )
        
        self.q_mean = nn.Linear(hidden_dim, latent_dim)
        self.q_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # ========================================
        # Decoder p(x | z) → ZINB parameters
        # ========================================
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # ZINB Parameters
        # Mean (μ): Expected count (must be positive)
        self.dec_mean = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Softplus()  # Ensure positive output
        )
        
        # Dispersion (θ): Controls variance (will be softplused in loss)
        self.dec_disp = nn.Linear(hidden_dim, input_dim)
        
        # Dropout probability (π): Zero-inflation logits (before sigmoid)
        self.dec_pi = nn.Linear(hidden_dim, input_dim)
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = μ + σ * ε, where ε ~ N(0, I)
        
        Args:
            mu (torch.Tensor): Mean of distribution [N, D]
            logvar (torch.Tensor): Log-variance of distribution [N, D]
        
        Returns:
            torch.Tensor: Sampled latent variable [N, D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode_prior(self, img, edge_index):
        """
        Compute prior distribution p(z | I, A) from image and graph
        
        Args:
            img (torch.Tensor): Image features [N, img_dim]
            edge_index (torch.Tensor): Graph edges [2, E]
        
        Returns:
            tuple: (μ_p, logvar_p)
        """
        # Encode image features
        h_i = self.img_encoder(img)
        
        # Apply GAT with spatial attention
        h_p = F.elu(self.p_gat_shared(h_i, edge_index))
        
        # Compute mean and log-variance
        mu_p = self.p_mean(h_p)
        logvar_p = self.p_logvar(h_p)
        
        return mu_p, logvar_p
    
    def encode_posterior(self, x, img, edge_index):
        """
        Compute posterior distribution q(z | X, I, A)
        
        Args:
            x (torch.Tensor): Gene expression [N, input_dim]
            img (torch.Tensor): Image features [N, img_dim]
            edge_index (torch.Tensor): Graph edges [2, E]
        
        Returns:
            tuple: (μ_q, logvar_q)
        """
        # Encode both modalities
        h_g = self.gene_encoder(x)
        h_i = self.img_encoder(img)
        
        # Concatenate for multi-modal fusion
        h_joint = torch.cat([h_g, h_i], dim=1)
        
        # Apply GAT with spatial attention
        h_q = F.elu(self.q_gat_shared(h_joint, edge_index))
        
        # Compute mean and log-variance
        mu_q = self.q_mean(h_q)
        logvar_q = self.q_logvar(h_q)
        
        return mu_q, logvar_q
    
    def decode(self, z):
        """
        Decode latent variable to ZINB parameters
        
        Args:
            z (torch.Tensor): Latent representation [N, latent_dim]
        
        Returns:
            dict: ZINB parameters {'mean', 'disp', 'pi'}
        """
        h_dec = self.decoder(z)
        
        mean = self.dec_mean(h_dec)
        disp = self.dec_disp(h_dec)
        pi = self.dec_pi(h_dec)
        
        return {
            'mean': mean,
            'disp': disp,
            'pi': pi
        }
    
    def forward(self, x, img, edge_index):
        """
        Forward pass through the entire model
        
        Args:
            x (torch.Tensor): Gene expression [N, input_dim]
            img (torch.Tensor): Image features [N, img_dim]
            edge_index (torch.Tensor): Graph edges [2, E]
        
        Returns:
            dict: {
                'mean': ZINB mean [N, input_dim],
                'disp': ZINB dispersion [N, input_dim],
                'pi': ZINB dropout logits [N, input_dim],
                'z': Sampled latent variable [N, latent_dim],
                'q_dist': (μ_q, logvar_q),
                'p_dist': (μ_p, logvar_p)
            }
        """
        # 1. Compute Prior Distribution p(z | I, A)
        mu_p, logvar_p = self.encode_prior(img, edge_index)
        
        # 2. Compute Posterior Distribution q(z | X, I, A)
        mu_q, logvar_q = self.encode_posterior(x, img, edge_index)
        
        # 3. Sample latent variable from posterior (reparameterization)
        z = self.reparameterize(mu_q, logvar_q)
        
        # 4. Decode to ZINB parameters
        zinb_params = self.decode(z)
        
        # 5. Return all outputs for loss computation
        return {
            'mean': zinb_params['mean'],
            'disp': zinb_params['disp'],
            'pi': zinb_params['pi'],
            'z': z,
            'q_dist': (mu_q, logvar_q),
            'p_dist': (mu_p, logvar_p)
        }
    
    def get_latent_representation(self, x, img, edge_index, use_mean=True):
        """
        Extract latent representation for downstream analysis
        
        Args:
            x (torch.Tensor): Gene expression
            img (torch.Tensor): Image features
            edge_index (torch.Tensor): Graph edges
            use_mean (bool): If True, return μ_q; else, sample from q(z)
        
        Returns:
            torch.Tensor: Latent representation [N, latent_dim]
        """
        mu_q, logvar_q = self.encode_posterior(x, img, edge_index)
        
        if use_mean:
            return mu_q
        else:
            return self.reparameterize(mu_q, logvar_q)
