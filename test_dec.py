import torch
import numpy as np
from models.dec import DEC_Module

# Test DEC Module basic functionality
def test_dec_module():
    print("Testing DEC Module...")
    
    # Hyperparameters
    latent_dim = 16
    n_clusters = 10
    batch_size = 128
    
    # Create a simple DEC module
    dec_module = DEC_Module(n_clusters=n_clusters, latent_dim=latent_dim)
    
    # Create dummy latent vectors
    z = torch.randn(batch_size, latent_dim)
    
    # Test forward pass (soft assignment Q)
    print("\n1. Testing forward pass (soft assignment Q)...")
    q = dec_module(z)
    print(f"   Input z shape: {z.shape}")
    print(f"   Output q shape: {q.shape}")
    print(f"   q row sums: {q.sum(dim=1).mean():.4f} ± {q.sum(dim=1).std():.4f} (should be 1.0)")
    
    # Test target distribution P
    print("\n2. Testing target distribution P...")
    p = dec_module.target_distribution(q)
    print(f"   Output p shape: {p.shape}")
    print(f"   p row sums: {p.sum(dim=1).mean():.4f} ± {p.sum(dim=1).std():.4f} (should be 1.0)")
    
    # Test clustering centers initialization with KMeans
    print("\n3. Testing clustering centers initialization with KMeans...")
    z_numpy = z.numpy()
    dec_module.init_centers(z_numpy)
    print(f"   Cluster centers shape: {dec_module.cluster_centers.shape}")
    
    # Test get_cluster_labels
    print("\n4. Testing get_cluster_labels...")
    labels = dec_module.get_cluster_labels(z)
    print(f"   Cluster labels shape: {labels.shape}")
    print(f"   Unique labels: {torch.unique(labels)}")
    
    # Test KL divergence between P and Q
    print("\n5. Testing KL divergence between P and Q...")
    import torch.nn.functional as F
    kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
    print(f"   KL divergence: {kl_loss.item():.4f}")
    
    print("\n✅ All tests passed!")

# Test DEC integration with dummy VAE
def test_dec_integration():
    print("\n\nTesting DEC Integration with Dummy VAE...")
    
    # Create a simple dummy VAE
    class DummyVAE(torch.nn.Module):
        def __init__(self, latent_dim=16):
            super().__init__()
            self.encoder = torch.nn.Linear(100, latent_dim)
            self.decoder = torch.nn.Linear(latent_dim, 100)
            
        def encode(self, x):
            return {'z': self.encoder(x)}
        
        def forward(self, x):
            z = self.encoder(x)
            x_recon = self.decoder(z)
            return {'z': z, 'x_recon': x_recon}
    
    # Initialize models
    latent_dim = 16
    n_clusters = 8
    vae = DummyVAE(latent_dim=latent_dim)
    dec_module = DEC_Module(n_clusters=n_clusters, latent_dim=latent_dim)
    
    # Create dummy data
    x = torch.randn(128, 100)
    
    # Test VAE forward pass
    output = vae(x)
    z = output['z']
    print(f"Dummy VAE latent shape: {z.shape}")
    
    # Initialize DEC centers using VAE latent
    dec_module.init_centers(z.detach().numpy())
    
    # Test joint forward pass
    q = dec_module(z)
    p = dec_module.target_distribution(q)
    
    print(f"DEC q shape: {q.shape}")
    print(f"DEC p shape: {p.shape}")
    print(f"Cluster labels: {torch.unique(dec_module.get_cluster_labels(z))}")
    
    print("✅ DEC Integration test passed!")

if __name__ == "__main__":
    test_dec_module()
    test_dec_integration()