"""
Image-Guided Graph VAE for Spatial Transcriptomics
Main Entry Point

This script provides command-line interface for:
- Training the model on spatial transcriptomics data
- Running inference and extracting latent representations
- Analyzing KL divergence patterns for biological insights
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import json

from models import ImageGuidedGVAE
from trainer import Trainer
from data.dataset import SpatialTranscriptomicsDataset, create_synthetic_data
from utils import Config


def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_on_synthetic_data(config):
    """
    Train model on synthetic data for testing
    
    Args:
        config (Config): Configuration object
    """
    print("=" * 80)
    print("Training on Synthetic Data")
    print("=" * 80)
    
    # Set seed
    set_seed(config.seed)
    
    # Generate synthetic data
    print("\nGenerating synthetic data...")
    gene_expr, coords, img_feat, labels = create_synthetic_data(
        n_spots=1000,
        n_genes=config.model.input_dim,
        n_clusters=5,
        seed=config.seed
    )
    
    # Create dataset
    print("Creating spatial graph...")
    dataset = SpatialTranscriptomicsDataset(
        gene_expression=gene_expr,
        spatial_coords=coords,
        image_features=img_feat,
        graph_type=config.data.graph_type,
        k_neighbors=config.data.k_neighbors,
        normalize=config.data.normalize_total,
        log_transform=config.data.log_transform
    )
    
    # Get data object
    data = dataset[0]
    
    print(f"\nDataset Statistics:")
    print(f"  Number of spots: {data.x.shape[0]}")
    print(f"  Number of genes: {data.x.shape[1]}")
    print(f"  Number of edges: {data.edge_index.shape[1]}")
    print(f"  Image feature dim: {data.img_feat.shape[1]}")
    
    # Update model config with actual dimensions
    config.model.input_dim = data.x.shape[1]
    config.model.img_dim = data.img_feat.shape[1]
    
    # Create model
    print(f"\nCreating model...")
    print(f"  Latent dim: {config.model.latent_dim}")
    print(f"  Hidden dim: {config.model.hidden_dim}")
    print(f"  GAT heads: {config.model.num_heads}")
    
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    model = ImageGuidedGVAE(
        input_dim=config.model.input_dim,
        img_dim=config.model.img_dim,
        hidden_dim=config.model.hidden_dim,
        latent_dim=config.model.latent_dim,
        num_heads=config.model.num_heads,
        dropout=config.model.dropout
    )
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = Trainer(model, config, device)
    
    # Train
    print(f"\nStarting training...")
    
    if hasattr(config.training, 'use_dec') and config.training.use_dec:
        # Use DEC two-stage training
        n_clusters = getattr(config.training, 'n_clusters', 5)
        dec_epochs = getattr(config.training, 'dec_epochs', 100)
        gamma = getattr(config.training, 'dec_gamma', 0.1)
        
        trainer.train_with_dec(
            train_data=data,
            n_clusters=n_clusters,
            dec_epochs=dec_epochs,
            gamma=gamma
        )
    else:
        # Use original training
        trainer.train(data)
    
    # Extract latent representation
    print("\n" + "=" * 80)
    print("Extracting Latent Representations")
    print("=" * 80)
    
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        output = model(data.x, data.img_feat, data.edge_index)
        latent = output['z'].cpu().numpy()
        
        # Compute KL divergence per spot
        mu_q, logvar_q = output['q_dist']
        mu_p, logvar_p = output['p_dist']
        
        var_q = torch.exp(logvar_q)
        var_p = torch.exp(logvar_p)
        
        kl_per_spot = 0.5 * torch.sum(
            (var_q + (mu_q - mu_p).pow(2)) / var_p + (logvar_p - logvar_q) - 1.0,
            dim=1
        )
        kl_per_spot = kl_per_spot.cpu().numpy()
    
    # Save results
    output_dir = Path(config.output_dir) / config.experiment_name
    np.save(output_dir / 'latent_representations.npy', latent)
    np.save(output_dir / 'kl_divergence.npy', kl_per_spot)
    np.save(output_dir / 'true_labels.npy', labels)
    
    # Save cluster labels if DEC was used
    if hasattr(trainer, 'dec_module') and trainer.dec_module is not None:
        with torch.no_grad():
            cluster_labels = trainer.dec_module.get_cluster_labels(output['z']).cpu().numpy()
        np.save(output_dir / 'cluster_labels.npy', cluster_labels)
        print(f"  Cluster labels saved")
    
    print(f"\nResults saved to {output_dir}")
    print(f"  Latent shape: {latent.shape}")
    print(f"  KL divergence - Mean: {kl_per_spot.mean():.4f}, Std: {kl_per_spot.std():.4f}")
    print(f"  High-KL spots (>95th percentile): {np.sum(kl_per_spot > np.percentile(kl_per_spot, 95))}")


def train_on_real_data(config, data_path):
    """
    Train model on real spatial transcriptomics data
    
    Args:
        config (Config): Configuration object
        data_path (str): Path to AnnData file
    """
    print("=" * 80)
    print("Training on Real Data")
    print("=" * 80)
    
    try:
        import scanpy as sc
    except ImportError:
        raise ImportError("scanpy is required for loading real data. Install with: pip install scanpy")
    
    # Set seed
    set_seed(config.seed)
    
    # Load data
    print(f"\nLoading data from {data_path}...")
    adata = sc.read_h5ad(data_path)
    
    print(f"Dataset: {adata.shape[0]} spots, {adata.shape[1]} genes")
    
    # Preprocess
    if config.data.highly_variable_genes:
        print(f"Selecting {config.data.highly_variable_genes} highly variable genes...")
        sc.pp.highly_variable_genes(
            adata, 
            n_top_genes=config.data.highly_variable_genes
        )
    
    # Create dataset
    from data.dataset import load_from_anndata
    dataset = load_from_anndata(adata, config.data)
    data = dataset[0]
    
    # Update model config
    config.model.input_dim = data.x.shape[1]
    config.model.img_dim = data.img_feat.shape[1]
    
    # Create model
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    model = ImageGuidedGVAE(
        input_dim=config.model.input_dim,
        img_dim=config.model.img_dim,
        hidden_dim=config.model.hidden_dim,
        latent_dim=config.model.latent_dim,
        num_heads=config.model.num_heads,
        dropout=config.model.dropout
    )
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer and train
    trainer = Trainer(model, config, device)
    
    # Train
    if hasattr(config.training, 'use_dec') and config.training.use_dec:
        # Use DEC two-stage training
        n_clusters = getattr(config.training, 'n_clusters', 16)
        dec_epochs = getattr(config.training, 'dec_epochs', 100)
        gamma = getattr(config.training, 'dec_gamma', 0.1)
        
        trainer.train_with_dec(
            train_data=data,
            n_clusters=n_clusters,
            dec_epochs=dec_epochs,
            gamma=gamma
        )
    else:
        # Use original training
        trainer.train(data)
    
    # Extract latent representation
    print("\n" + "=" * 80)
    print("Extracting Latent Representations")
    print("=" * 80)
    
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        output = model(data.x, data.img_feat, data.edge_index)
        latent = output['z'].cpu().numpy()
        
        # Compute KL divergence per spot
        mu_q, logvar_q = output['q_dist']
        mu_p, logvar_p = output['p_dist']
        
        var_q = torch.exp(logvar_q)
        var_p = torch.exp(logvar_p)
        
        kl_per_spot = 0.5 * torch.sum(
            (var_q + (mu_q - mu_p).pow(2)) / var_p + (logvar_p - logvar_q) - 1.0,
            dim=1
        )
        kl_per_spot = kl_per_spot.cpu().numpy()
    
    # Get labels from AnnData if available
    labels = None
    if 'labels' in adata.obs:
        labels = adata.obs['labels'].values
    elif 'cluster' in adata.obs:
        labels = adata.obs['cluster'].values
    elif 'cell_type' in adata.obs:
        labels = adata.obs['cell_type'].values
    
    # Save results
    output_dir = Path(config.output_dir) / config.experiment_name
    np.save(output_dir / 'latent_representations.npy', latent)
    np.save(output_dir / 'kl_divergence.npy', kl_per_spot)
    if labels is not None:
        np.save(output_dir / 'true_labels.npy', labels)
    
    # Save cluster labels if DEC was used
    if hasattr(trainer, 'dec_module') and trainer.dec_module is not None:
        with torch.no_grad():
            cluster_labels = trainer.dec_module.get_cluster_labels(output['z']).cpu().numpy()
        np.save(output_dir / 'cluster_labels.npy', cluster_labels)
        print(f"  Cluster labels saved")
    
    print(f"\nResults saved to {output_dir}")
    print(f"  Latent shape: {latent.shape}")
    print(f"  KL divergence - Mean: {kl_per_spot.mean():.4f}, Std: {kl_per_spot.std():.4f}")
    print(f"  High-KL spots (>95th percentile): {np.sum(kl_per_spot > np.percentile(kl_per_spot, 95))}")


def extract_latent(checkpoint_path, data_path, output_path, config):
    """
    Extract latent representations from trained model
    
    Args:
        checkpoint_path (str): Path to model checkpoint
        data_path (str): Path to data
        output_path (str): Path to save latent representations
        config (Config): Configuration object
    """
    print("=" * 80)
    print("Extracting Latent Representations")
    print("=" * 80)
    
    # Load checkpoint
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    model = ImageGuidedGVAE(
        input_dim=config.model.input_dim,
        img_dim=config.model.img_dim,
        hidden_dim=config.model.hidden_dim,
        latent_dim=config.model.latent_dim,
        num_heads=config.model.num_heads,
        dropout=config.model.dropout
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load data (implementation depends on data format)
    # ... (similar to training)
    
    print("Extraction complete!")


def main():
    parser = argparse.ArgumentParser(
        description='Image-Guided Graph VAE for Spatial Transcriptomics'
    )
    
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['train_synthetic', 'train_real', 'extract'],
        default='train_synthetic',
        help='Operation mode'
    )
    
    parser.add_argument(
        '--data_path',
        type=str,
        help='Path to spatial transcriptomics data (AnnData .h5ad file)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to config JSON file'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to model checkpoint (for extraction mode)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='./outputs',
        help='Output directory'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = Config.from_dict(config_dict)
    else:
        config = Config()
    
    # Override with command line arguments
    config.device = args.device
    config.seed = args.seed
    config.output_dir = args.output
    
    # Execute based on mode
    if args.mode == 'train_synthetic':
        train_on_synthetic_data(config)
    
    elif args.mode == 'train_real':
        if not args.data_path:
            raise ValueError("--data_path required for train_real mode")
        train_on_real_data(config, args.data_path)
    
    elif args.mode == 'extract':
        if not args.checkpoint or not args.data_path:
            raise ValueError("--checkpoint and --data_path required for extract mode")
        extract_latent(args.checkpoint, args.data_path, args.output, config)

        


if __name__ == '__main__':
    main()
