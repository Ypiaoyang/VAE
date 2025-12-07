"""
Example script demonstrating how to use DEC (Deep Embedded Clustering) module
in the Image-Guided GVAE model for spatial transcriptomics clustering.
"""

import torch
import numpy as np
from utils.config import Config, TrainingConfig
from models import ImageGuidedGVAE, ZINBLoss
from trainer import Trainer
from torch_geometric.data import Data

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def create_dummy_data(num_spots=1000, num_genes=2000, img_dim=1024, latent_dim=16):
    """Create dummy spatial transcriptomics data for testing."""
    # Create dummy gene expression data (counts)
    x_counts = torch.randint(0, 100, (num_spots, num_genes), dtype=torch.float32)
    
    # Normalized expression data (simulating preprocessing)
    x = torch.log1p(x_counts)
    
    # Create dummy image features
    img_feat = torch.randn(num_spots, img_dim)
    
    # Create dummy spatial graph (knn)
    edge_index = torch.randint(0, num_spots, (2, num_spots * 6), dtype=torch.long)
    
    # Create scale factor for ZINB loss (total counts per spot)
    scale_factor = torch.sum(x_counts, dim=1, keepdim=True)
    
    # Create PyG Data object with both raw counts and normalized data
    data = Data(
        x=x,                # Normalized expression data
        x_counts=x_counts,  # Raw counts (required for ZINB loss)
        img_feat=img_feat,  # Image features
        edge_index=edge_index,  # Spatial graph
        scale_factor=scale_factor  # Scale factor for ZINB loss
    )
    
    return data

def main():
    # 1. Configure the model with DEC enabled
    print("\n" + "=" * 80)
    print("CONFIGURING MODEL WITH DEC")
    print("=" * 80)
    
    # Create base config
    config = Config()
    
    # Enable DEC in training config
    config.training.use_dec = True
    config.training.n_clusters = 12  # Set number of clusters
    config.training.dec_epochs = 100  # Set DEC fine-tuning epochs
    config.training.dec_gamma = 0.1   # Set DEC loss weight
    
    # Adjust other parameters if needed
    config.model.latent_dim = 16  # Lower latent dim is better for clustering
    config.training.epochs = 500  # VAE pretraining epochs
    
    print("Model Configuration:")
    print(f"  - Latent dimension: {config.model.latent_dim}")
    print(f"  - DEC enabled: {config.training.use_dec}")
    print(f"  - Number of clusters: {config.training.n_clusters}")
    print(f"  - DEC epochs: {config.training.dec_epochs}")
    print(f"  - DEC loss weight: {config.training.dec_gamma}")
    
    # 2. Create dummy data
    print("\n" + "=" * 80)
    print("CREATING DUMMY DATA")
    print("=" * 80)
    
    data = create_dummy_data(
        num_spots=1000,
        num_genes=2000,
        img_dim=1024,
        latent_dim=config.model.latent_dim
    )
    
    print(f"Created dummy data with:")
    print(f"  - {data.num_nodes} spots")
    print(f"  - {data.x.shape[1]} genes")
    print(f"  - {data.edge_index.shape[1]} edges")
    
    # 3. Initialize model
    print("\n" + "=" * 80)
    print("INITIALIZING MODEL")
    print("=" * 80)
    
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model with parameters from config
    model = ImageGuidedGVAE(
        input_dim=config.model.input_dim,
        img_dim=config.model.img_dim,
        hidden_dim=config.model.hidden_dim,
        latent_dim=config.model.latent_dim,
        num_heads=config.model.num_heads,
        dropout=config.model.dropout
    ).to(device)
    data = data.to(device)
    
    print("Model initialized successfully!")
    
    # 4. Train with DEC
    print("\n" + "=" * 80)
    print("STARTING TRAINING WITH DEC")
    print("=" * 80)
    
    trainer = Trainer(model, config, device)
    
    # The trainer will automatically:
    # 1. Pre-train the VAE (config.training.epochs)
    # 2. Initialize DEC clustering centers with KMeans
    # 3. Fine-tune with DEC loss (config.training.dec_epochs)
    trainer.train_with_dec(
        train_data=data,
        n_clusters=config.training.n_clusters,
        dec_epochs=config.training.dec_epochs,
        gamma=config.training.dec_gamma
    )
    
    # 5. Evaluate clustering results
    print("\n" + "=" * 80)
    print("EVALUATING CLUSTERING RESULTS")
    print("=" * 80)
    
    model.eval()
    with torch.no_grad():
        output = model(data.x, data.img_feat, data.edge_index)
        latent = output['z']
        
        # Get cluster labels from DEC module
        cluster_labels = trainer.dec_module.get_cluster_labels(latent)
        unique_labels = torch.unique(cluster_labels)
        
        print(f"Clustering results:")
        print(f"  - Unique clusters: {len(unique_labels)}")
        print(f"  - Cluster counts: {torch.bincount(cluster_labels)}")
        
        # Compute clustering statistics
        print(f"\nCluster distribution:")
        for label in unique_labels:
            count = (cluster_labels == label).sum().item()
            percentage = (count / len(cluster_labels)) * 100
            print(f"  Cluster {label}: {count} spots ({percentage:.1f}%)")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()