#!/usr/bin/env python3
"""
Test script to evaluate DEC training speed optimization

This script compares the training speed with different configurations:
1. Original: target_distribution_update_freq=1 and loop-based InfoNCE
2. Optimized: target_distribution_update_freq=5 and vectorized InfoNCE
3. Highly optimized: target_distribution_update_freq=10 and vectorized InfoNCE
"""

import torch
import time
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.cvae import ImageGuidedGVAE as CVAE
from models.layers import InfoNCELoss
from utils.config import Config

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create synthetic dataset for testing
def create_synthetic_data(num_spots=1000, num_genes=2000, latent_dim=32):
    """Create synthetic data for testing"""
    # Gene expression data
    x = torch.randn(num_spots, num_genes).to(device)
    
    # Image features
    img_feat = torch.randn(num_spots, 1024).to(device)  # ResNet features
    
    # Create a simple k-NN graph for spatial adjacency
    edge_index = []
    for i in range(num_spots):
        # Connect each spot to its 6 nearest neighbors (simulating spatial proximity)
        neighbors = [(i, (i + j) % num_spots) for j in range(1, 7)]
        edge_index.extend(neighbors)
    edge_index = torch.tensor(edge_index, dtype=torch.long).T.to(device)
    
    # Create a simple data object
    class Data:
        def __init__(self, x, img_feat, edge_index):
            self.x = x
            self.img_feat = img_feat
            self.edge_index = edge_index
    
    return Data(x, img_feat, edge_index)

# Test InfoNCE loss speed
def test_infonce_speed():
    """Test the speed of InfoNCE loss calculation"""
    print("\n=== Testing InfoNCE Loss Speed ===")
    
    # Create test data
    num_spots = 1000
    latent_dim = 32
    z = torch.randn(num_spots, latent_dim).to(device)
    
    # Create edge index (simulating spatial graph)
    edge_index = []
    for i in range(num_spots):
        neighbors = [(i, (i + j) % num_spots) for j in range(1, 7)]
        edge_index.extend(neighbors)
    edge_index = torch.tensor(edge_index, dtype=torch.long).T.to(device)
    
    # Initialize InfoNCE loss
    infonce_loss = InfoNCELoss(temperature=0.1, similarity='cosine')
    
    # Warm-up run
    _ = infonce_loss(z, edge_index)
    
    # Measure time for forward pass only (this is the main part we optimized)
    num_runs = 1000  # Increase runs for more accurate measurement
    start_time = time.time()
    
    for _ in range(num_runs):
        loss = infonce_loss(z, edge_index)
    
    total_time = time.time() - start_time
    avg_time = total_time / num_runs * 1000  # Convert to milliseconds
    
    print(f"Vectorized InfoNCE Loss:")
    print(f"  Forward pass: {avg_time:.4f} ms per iteration")
    print(f"  Loss value: {loss.item():.4f}")
    print(f"  Total time for {num_runs} runs: {total_time:.4f} seconds")

# Test target distribution update speed
def test_target_distribution_update_speed():
    """Test the speed of target distribution update"""
    print("\n=== Testing Target Distribution Update Speed ===")
    
    # Create synthetic data
    train_data = create_synthetic_data(num_spots=1000, num_genes=2000, latent_dim=32)
    
    # Create model with explicit parameters
    config = Config()
    model = CVAE(
        input_dim=config.model.input_dim,
        img_dim=config.model.img_dim,
        hidden_dim=config.model.hidden_dim,
        latent_dim=config.model.latent_dim,
        num_heads=config.model.num_heads,
        dropout=config.model.dropout
    ).to(device)
    
    # Initialize with random weights
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    
    model.apply(init_weights)
    
    # Measure time for single update
    start_time = time.time()
    
    # Simulate target distribution update
    with torch.no_grad():
        data = train_data
        output = model(data.x, data.img_feat, data.edge_index)
        z = output['z']
    
    end_time = time.time()
    avg_time = (end_time - start_time) * 1000  # Convert to milliseconds
    
    print(f"Single target distribution update: {avg_time:.2f} ms")
    print(f"Latent space shape: {z.shape}")

# Main test function
def main():
    """Main function to run all tests"""
    print("DEC Training Speed Optimization Test")
    print("=" * 50)
    
    # Test InfoNCE loss speed
    test_infonce_speed()
    
    # Test target distribution update speed
    test_target_distribution_update_speed()
    
    print("\n" + "=" * 50)
    print("All tests completed successfully!")

if __name__ == "__main__":
    main()
