import torch
import numpy as np
from utils.config import Config
from models import ImageGuidedGVAE
from trainer import Trainer
from data.dataset import create_synthetic_data, SpatialTranscriptomicsDataset

def test_training_save():
    """Test if training saves model to timestamped directory"""
    print("Testing training with timestamped save directory...")
    
    # Create config
    config = Config()
    config.training.epochs = 2  # Short training for test
    config.training.save_interval = 1
    
    # Generate synthetic data
    gene_expr, coords, img_feat, labels = create_synthetic_data(
        n_spots=100, n_genes=100, n_clusters=2, seed=42
    )
    
    # Create dataset
    dataset = SpatialTranscriptomicsDataset(
        gene_expression=gene_expr,
        spatial_coords=coords,
        image_features=img_feat,
        graph_type="knn",
        k_neighbors=6
    )
    data = dataset[0]
    
    # Update model config
    config.model.input_dim = data.x.shape[1]
    config.model.img_dim = data.img_feat.shape[1]
    
    # Create model and trainer
    device = torch.device("cpu")
    model = ImageGuidedGVAE(
        input_dim=config.model.input_dim,
        img_dim=config.model.img_dim,
        hidden_dim=64,  # Smaller for quick test
        latent_dim=16,
        num_heads=2,
        dropout=0.1
    )
    
    trainer = Trainer(model, config, device)
    print(f"Output directory: {trainer.output_dir}")
    print(f"Expected: ./outputs/image_guided_gvae_YYYYMMDD_HHMMSS")
    
    # Start training
    trainer.train(data)
    
    # Check if directory exists
    import os
    assert os.path.exists(trainer.output_dir), f"Output directory {trainer.output_dir} does not exist!"
    
    # Check if checkpoint files exist
    checkpoints = [f for f in os.listdir(trainer.output_dir) if f.endswith(".pt")]
    assert len(checkpoints) > 0, f"No checkpoint files found in {trainer.output_dir}!"
    
    print(f"✓ Test passed! Model saved to: {trainer.output_dir}")
    print(f"✓ Checkpoint files: {checkpoints}")

if __name__ == "__main__":
    test_training_save()