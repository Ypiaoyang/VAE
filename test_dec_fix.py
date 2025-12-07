"""
Test script to verify the DEC module fixes:
1. Reduced DEC weight from 0.1 to 0.01
2. Lower learning rate for DEC fine-tuning
3. Stable target distribution P update
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.config import Config
from models import ImageGuidedGVAE, DEC_Module
from trainer import Trainer
from torch_geometric.data import Data
from sklearn.metrics import silhouette_score, davies_bouldin_score

def create_dummy_data(num_spots=500, num_genes=2000, img_dim=1024, latent_dim=16):
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

def test_dec_fix_comparison():
    """Compare DEC performance before and after fixes."""
    print("\n" + "=" * 80)
    print("TESTING DEC MODULE FIXES")
    print("=" * 80)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create dummy data
    data = create_dummy_data(num_spots=500, latent_dim=16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    
    # Test configurations
    test_configs = [
        {
            "name": "Before Fix",
            "dec_gamma": 0.1,
            "dec_lr": None,  # Uses default LR (not reduced)
            "description": "DEC weight=0.1, LR=same as pre-training"
        },
        {
            "name": "After Fix",
            "dec_gamma": 0.01,
            "dec_lr": None,  # Uses 1/10 of pre-training LR
            "description": "DEC weight=0.01, LR=1/10 of pre-training"
        }
    ]
    
    results = []
    
    for config in test_configs:
        print(f"\n" + "=" * 60)
        print(f"Testing: {config['name']}")
        print(f"Description: {config['description']}")
        print("=" * 60)
        
        # Create model config
        model_config = Config()
        model_config.model.latent_dim = 16
        model_config.training.epochs = 200  # Reduced for faster testing
        model_config.training.dec_epochs = 50
        model_config.training.dec_gamma = config['dec_gamma']
        if config['dec_lr'] is not None:
            model_config.training.dec_learning_rate = config['dec_lr']
        
        # Initialize model
        model = ImageGuidedGVAE(
            input_dim=model_config.model.input_dim,
            img_dim=model_config.model.img_dim,
            hidden_dim=model_config.model.hidden_dim,
            latent_dim=model_config.model.latent_dim,
            num_heads=model_config.model.num_heads,
            dropout=model_config.model.dropout
        ).to(device)
        
        # Create trainer
        trainer = Trainer(model, model_config, device)
        
        # Train with DEC
        trainer.train_with_dec(
            train_data=data,
            n_clusters=8,
            dec_epochs=model_config.training.dec_epochs,
            gamma=model_config.training.dec_gamma
        )
        
        # Evaluate clustering results
        model.eval()
        with torch.no_grad():
            output = model(data.x, data.img_feat, data.edge_index)
            latent = output['z'].cpu().numpy()
            cluster_labels = trainer.dec_module.get_cluster_labels(output['z']).cpu().numpy()
        
        # Calculate clustering metrics
        silhouette_avg = silhouette_score(latent, cluster_labels)
        davies_bouldin = davies_bouldin_score(latent, cluster_labels)
        
        print(f"\nClustering Metrics:")
        print(f"  Silhouette Score: {silhouette_avg:.4f}")
        print(f"  Davies-Bouldin Index: {davies_bouldin:.4f}")
        print(f"  Unique Clusters: {len(np.unique(cluster_labels))}")
        print(f"  Cluster Distribution: {np.bincount(cluster_labels)}")
        
        # Store results
        results.append({
            "name": config['name'],
            "silhouette": silhouette_avg,
            "davies_bouldin": davies_bouldin,
            "latent": latent,
            "labels": cluster_labels
        })
    
    # Visualize comparison
    visualize_results(results)
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    
    # Summary
    print("\nSummary of Results:")
    for result in results:
        print(f"{result['name']}:")
        print(f"  Silhouette Score: {result['silhouette']:.4f}")
        print(f"  Davies-Bouldin Index: {result['davies_bouldin']:.4f}")
    
    # Determine if fix improved results
    before = next(r for r in results if r['name'] == 'Before Fix')
    after = next(r for r in results if r['name'] == 'After Fix')
    
    print("\nConclusion:")
    if after['silhouette'] > before['silhouette'] and after['davies_bouldin'] < before['davies_bouldin']:
        print("✅ DEC fixes improved clustering performance!")
        print(f"   Silhouette increased by: {after['silhouette'] - before['silhouette']:.4f}")
        print(f"   Davies-Bouldin decreased by: {before['davies_bouldin'] - after['davies_bouldin']:.4f}")
    else:
        print("❌ DEC fixes did not improve clustering performance.")

def visualize_results(results):
    """Visualize clustering results comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    for i, result in enumerate(results):
        # Plot 1: Latent space visualization
        ax1 = axes[i, 0]
        scatter = ax1.scatter(result['latent'][:, 0], result['latent'][:, 1], 
                            c=result['labels'], cmap='viridis', s=20, alpha=0.7)
        ax1.set_title(f'{result["name"]}: Latent Space')
        ax1.set_xlabel('Latent Dimension 1')
        ax1.set_ylabel('Latent Dimension 2')
        plt.colorbar(scatter, ax=ax1)
        
        # Plot 2: Cluster distribution
        ax2 = axes[i, 1]
        unique, counts = np.unique(result['labels'], return_counts=True)
        ax2.bar(unique, counts, color='skyblue', edgecolor='black')
        ax2.set_title(f'{result["name"]}: Cluster Distribution')
        ax2.set_xlabel('Cluster ID')
        ax2.set_ylabel('Number of Spots')
        ax2.set_xticks(unique)
        
        # Add counts on top of bars
        for u, count in zip(unique, counts):
            ax2.text(u, count + 2, str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('dec_fix_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to 'dec_fix_comparison.png'")

if __name__ == "__main__":
    test_dec_fix_comparison()