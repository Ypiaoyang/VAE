"""
Custom Layers and Loss Functions

This module implements the Zero-Inflated Negative Binomial (ZINB) loss function
with numerical stability for spatial transcriptomics data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ZINBLoss(nn.Module):
    """
    Zero-Inflated Negative Binomial Loss
    
    Models gene expression counts with extreme sparsity using ZINB distribution.
    
    Mathematical formulation:
    P(X=x | μ, θ, π) = {
        π + (1-π)(θ/(θ+μ))^θ                                    if x = 0
        (1-π) * Γ(x+θ)/(x!Γ(θ)) * (θ/(θ+μ))^θ * (μ/(θ+μ))^x   if x > 0
    }
    
    Parameters:
        eps (float): Small constant for numerical stability
        scale_factor_mode (str): How to apply library size normalization
                                'mean' - divide by mean library size
                                'spot' - use per-spot scale factors
    """
    
    def __init__(self, eps=1e-10, scale_factor_mode='mean'):
        super().__init__()
        self.eps = eps
        self.scale_factor_mode = scale_factor_mode
    
    def forward(self, x, mean, disp, pi, scale_factor=1.0):
        """
        Compute ZINB negative log-likelihood
        
        Args:
            x (torch.Tensor): Observed counts [N, G] (ground truth)
            mean (torch.Tensor): Predicted mean [N, G] (from decoder)
            disp (torch.Tensor): Predicted dispersion [N, G] or [G] (theta)
            pi (torch.Tensor): Predicted dropout logits [N, G] (before sigmoid)
            scale_factor (torch.Tensor or float): Library size normalization [N, 1] or scalar
        
        Returns:
            torch.Tensor: Mean negative log-likelihood across all observations
        """
        eps = self.eps
        
        # Apply library size normalization (sequencing depth correction)
        # mean represents the expected count before normalization
        mean = mean * scale_factor
        
        # Ensure dispersion is positive using softplus
        # disp is the inverse of dispersion in NB parameterization
        disp = F.softplus(disp) + eps
        
        # Compute Negative Binomial log-likelihood component
        # Using the parameterization: NB(r, p) where r=theta, p=theta/(theta+mu)
        
        # Log-Gamma terms: log[Γ(x+θ)] - log[Γ(θ)] - log[Γ(x+1)]
        # Using lgamma for numerical stability
        t1 = (
            torch.lgamma(x + disp + eps) 
            - torch.lgamma(disp + eps) 
            - torch.lgamma(x + 1.0 + eps)
        )
        
        # θ * log(θ/(θ+μ))
        t2 = disp * (torch.log(disp + eps) - torch.log(disp + mean + eps))
        
        # x * log(μ/(θ+μ))
        t3 = x * (torch.log(mean + eps) - torch.log(disp + mean + eps))
        
        # Complete NB log-probability
        nb_log_prob = t1 + t2 + t3
        
        # Zero-Inflation component
        # π represents dropout probability
        # We use log-sigmoid for numerical stability
        
        # Compute log(1 + exp(-pi)) = softplus(-pi)
        softplus_pi = F.softplus(-pi)
        
        # log(sigmoid(pi)) = log(π_prob) = -pi - softplus(-pi)
        log_theta_pi = -pi - softplus_pi
        
        # log(1 - sigmoid(pi)) = log(1 - π_prob) = -softplus(-pi)
        log_1_minus_pi = -softplus_pi
        
        # Case 1: x = 0
        # P(X=0) = π + (1-π) * NB(0)
        # In log space: log[exp(log_π) + exp(log_(1-π) + log_NB(0))]
        # Use logsumexp for numerical stability
        zero_nb = disp * (torch.log(disp + eps) - torch.log(disp + mean + eps))
        zero_case = torch.logsumexp(
            torch.stack([log_theta_pi, log_1_minus_pi + zero_nb]), 
            dim=0
        )
        
        # Case 2: x > 0
        # P(X=x) = (1-π) * NB(x)
        non_zero_case = log_1_minus_pi + nb_log_prob
        
        # Combine using mask
        # Create mask for zero vs non-zero counts
        mask = (x < eps).float()  # 1 if x==0, 0 otherwise
        
        # Select appropriate log-probability
        log_lik = mask * zero_case + (1.0 - mask) * non_zero_case
        
        # Return mean negative log-likelihood
        # Sum over genes (dim=1), then mean over spots
        return -torch.mean(torch.sum(log_lik, dim=1))


class GaussianKLDivergence(nn.Module):
    """
    KL Divergence between two multivariate Gaussian distributions
    
    D_KL(N(μ_q, Σ_q) || N(μ_p, Σ_p))
    
    Assumes diagonal covariance matrices (factorized Gaussian).
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, mu_q, logvar_q, mu_p, logvar_p):
        """
        Compute KL divergence between two Gaussians
        
        Args:
            mu_q (torch.Tensor): Mean of posterior [N, D]
            logvar_q (torch.Tensor): Log-variance of posterior [N, D]
            mu_p (torch.Tensor): Mean of prior [N, D]
            logvar_p (torch.Tensor): Log-variance of prior [N, D]
        
        Returns:
            torch.Tensor: KL divergence summed over dimensions and spots
        
        Formula:
            KL = 0.5 * Σ_d [ (σ²_q + (μ_q - μ_p)²) / σ²_p + log(σ²_p / σ²_q) - 1 ]
        """
        # Convert log-variance to variance
        var_q = torch.exp(logvar_q)
        var_p = torch.exp(logvar_p)
        
        # Compute KL divergence element-wise
        # Term 1: (σ²_q + (μ_q - μ_p)²) / σ²_p
        kl_element = (var_q + (mu_q - mu_p).pow(2)) / var_p
        
        # Term 2: log(σ²_p / σ²_q) = logvar_p - logvar_q
        kl_element = kl_element + (logvar_p - logvar_q)
        
        # Term 3: -1
        kl_element = kl_element - 1.0
        
        # Multiply by 0.5 and sum over all dimensions and spots
        kl_div = 0.5 * torch.sum(kl_element)
        
        return kl_div


class InfoNCELoss(nn.Module):
    """
    InfoNCE Loss for Contrastive Learning
    
    Implements the InfoNCE loss function:
    L_contrast = -log [ exp(sim(z_i, z_pos)/τ) / sum(exp(sim(z_i, z_neg)/τ)) ]
    
    Where:
        - z_i is the latent representation of spot i
        - z_pos is the latent representation of a spatially adjacent spot (positive sample)
        - z_neg is the latent representation of a non-adjacent spot (negative sample)
        - τ is the temperature parameter
        - sim is a similarity metric (e.g., cosine similarity)
    
    Args:
        temperature (float): Temperature parameter for softmax
        similarity (str): Similarity metric ('cosine' or 'euclidean')
    """
    
    def __init__(self, temperature=0.1, similarity='cosine'):
        super().__init__()
        self.temperature = temperature
        self.similarity = similarity
    
    def forward(self, z, edge_index):
        """
        Compute InfoNCE loss using spatial adjacency information
        
        Args:
            z (torch.Tensor): Latent representations [N, D]
            edge_index (torch.Tensor): Spatial adjacency graph [2, E]
        
        Returns:
            torch.Tensor: Mean InfoNCE loss across all spots
        """
        N = z.size(0)  # Number of spots
        
        # Build adjacency matrix from edge_index
        pos_mask = torch.zeros((N, N), device=z.device)
        pos_mask[edge_index[0], edge_index[1]] = 1
        
        # Include self as positive for stability (avoid division by zero)
        torch.diagonal(pos_mask).fill_(1)
        
        # Compute similarity matrix
        if self.similarity == 'cosine':
            # Normalize vectors for cosine similarity
            z_normalized = F.normalize(z, p=2, dim=1)
            sim_matrix = torch.matmul(z_normalized, z_normalized.T) / self.temperature
        elif self.similarity == 'euclidean':
            # Euclidean distance converted to similarity
            dist_matrix = torch.cdist(z, z, p=2)
            sim_matrix = -dist_matrix / self.temperature
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity}")
        
        # Compute exponentials of similarities
        exp_sim = torch.exp(sim_matrix)
        
        # Numerator: sum of exponentials for positive pairs (spatially adjacent or self)
        numerator = (exp_sim * pos_mask).sum(dim=1)
        
        # Denominator: sum of exponentials for all pairs
        denominator = exp_sim.sum(dim=1)
        
        # Compute InfoNCE loss for each spot and take mean
        loss = -torch.log(numerator / denominator).mean()
        
        return loss
