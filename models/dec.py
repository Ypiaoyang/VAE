import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans


class DEC_Module(nn.Module):
    """
    Deep Embedded Clustering (DEC) Module
    
    This module implements the DEC algorithm which adds a clustering loss
    to the VAE, helping to create more compact and well-separated clusters
    in the latent space.
    
    Args:
        n_clusters (int): Number of clusters to identify
        latent_dim (int): Dimensionality of the latent space
        alpha (float, optional): Degrees of freedom for Student's t-distribution (default: 1.0)
    """
    def __init__(self, n_clusters, latent_dim, alpha=1.0):
        super(DEC_Module, self).__init__()
        self.alpha = alpha
        # Cluster centers are learnable parameters
        self.cluster_centers = nn.Parameter(torch.Tensor(n_clusters, latent_dim))
        
        # Initialize cluster centers (will be overwritten by KMeans)
        nn.init.xavier_normal_(self.cluster_centers.data)

    def init_centers(self, z_numpy):
        """
        Initialize cluster centers using KMeans on pre-trained latent representations
        
        Args:
            z_numpy (numpy.ndarray): Latent representations from pre-trained VAE
        """
        print(f"DEC: Initializing {self.cluster_centers.shape[0]} cluster centers with KMeans...")
        kmeans = KMeans(n_clusters=self.cluster_centers.shape[0], n_init=20, random_state=42)
        kmeans.fit(z_numpy)
        self.cluster_centers.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float).to(self.cluster_centers.device)
        
        # Return initial cluster labels for reference
        return kmeans.labels_

    def target_distribution(self, q):
        """
        Compute target distribution P from soft assignment Q
        
        Args:
            q (torch.Tensor): Soft assignment matrix [N, n_clusters]
            
        Returns:
            torch.Tensor: Target distribution matrix [N, n_clusters]
        """
        weight = q**2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()

    def forward(self, z):
        """
        Compute soft assignment distribution Q using Student's t-distribution
        
        Args:
            z (torch.Tensor): Latent representations [N, latent_dim]
            
        Returns:
            torch.Tensor: Soft assignment matrix [N, n_clusters]
        """
        # Compute squared Euclidean distance between z and cluster centers
        # Using efficient broadcasting and matrix operations
        x_sq = torch.sum(z**2, dim=1, keepdim=True)
        c_sq = torch.sum(self.cluster_centers**2, dim=1)
        dist_sq = x_sq + c_sq - 2 * torch.matmul(z, self.cluster_centers.t())
        
        # Apply Student-t distribution formula
        q = torch.pow(1.0 + dist_sq / self.alpha, -(self.alpha + 1.0) / 2.0)
        q = q / torch.sum(q, dim=1, keepdim=True)  # Normalize to get probabilities
        return q

    def get_cluster_labels(self, z):
        """
        Get hard cluster assignments from latent representations
        
        Args:
            z (torch.Tensor): Latent representations [N, latent_dim]
            
        Returns:
            torch.Tensor: Cluster labels [N]
        """
        q = self.forward(z)
        return torch.argmax(q, dim=1)
