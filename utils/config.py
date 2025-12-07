"""
Configuration Management

Centralized configuration for model, training, and data parameters.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    input_dim: int = 2000  # Number of genes (will be set by dataset)
    img_dim: int = 1024    # Image feature dimension (e.g., ResNet/UNI)
    hidden_dim: int = 256  # Hidden layer size
    latent_dim: int = 32   # Latent space dimensionality
    num_heads: int = 3     # GAT attention heads
    dropout: float = 0.1   # Dropout probability


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 512  # Number of spots per batch
    epochs: int = 1000
    
    # KL Annealing parameters
    kl_annealing_start: int = 100     # Epoch to start KL annealing
    kl_annealing_end: int = 400      # Epoch to reach full KL weight
    kl_weight_max: float = 1.0      # Maximum KL weight (beta in beta-VAE)
    
    # Regularization
    gradient_clip: float = 5.0      # Gradient clipping threshold
    
    # Logging
    log_interval: int = 10          # Log every N epochs
    save_interval: int = 50         # Save checkpoint every N epochs
    
    # DEC (Deep Embedded Clustering) parameters
    use_dec: bool = False           # Whether to use DEC module for clustering
    n_clusters: int = 16            # Number of clustering centers
    dec_epochs: int = 200           # Number of DEC fine-tuning epochs
    dec_gamma: float = 0.01         # Weight for DEC loss (reduced from 0.1 to 0.01)
    dec_learning_rate: Optional[float] = None  # Learning rate for DEC fine-tuning (default: 1/10 of pre-training LR)
    target_distribution_update_freq: int = 1   # Update target distribution every N epochs (1 = every epoch)
    
    # InfoNCE Loss for Contrastive Learning
    use_infonce: bool = False        # Whether to use InfoNCE loss for contrastive learning
    infonce_weight: float = 0.1     # Weight for InfoNCE loss
    infonce_temperature: float = 0.1  # Temperature parameter for InfoNCE loss
    infonce_similarity: str = "cosine"  # Similarity metric ("cosine" or "euclidean")


@dataclass
class DataConfig:
    """Data processing configuration"""
    data_path: Optional[str] = None
    image_path: Optional[str] = None
    
    # Preprocessing
    normalize_total: bool = True    # Total count normalization
    log_transform: bool = True      # Log(1+x) transformation
    highly_variable_genes: Optional[int] = 2000  # Select top N HVGs
    
    # Spatial graph construction
    graph_type: str = "knn"         # "knn" or "radius"
    k_neighbors: int = 6            # For k-NN graph
    radius: Optional[float] = None  # For radius graph
    
    # Image feature extraction
    image_encoder: str = "resnet"   # "resnet", "uni", or "custom"
    image_size: int = 224           # Input image size for encoder
    
    # Data split
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1


@dataclass
class Config:
    """Complete configuration"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # System
    device: str = "cuda"  # "cuda" or "cpu"
    seed: int = 42
    num_workers: int = 4
    
    # Output
    output_dir: str = "./outputs"
    experiment_name: str = "image_guided_gvae"
    
    def __post_init__(self):
        """Validate configuration"""
        assert self.data.train_ratio + self.data.val_ratio + self.data.test_ratio == 1.0
        assert self.training.kl_annealing_start < self.training.kl_annealing_end
        assert 0.0 <= self.training.kl_weight_max <= 1.0
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary"""
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            data=DataConfig(**config_dict.get('data', {})),
            **{k: v for k, v in config_dict.items() 
               if k not in ['model', 'training', 'data']}
        )
    
    def to_dict(self):
        """Convert config to dictionary"""
        return {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'data': self.data.__dict__,
            'device': self.device,
            'seed': self.seed,
            'num_workers': self.num_workers,
            'output_dir': self.output_dir,
            'experiment_name': self.experiment_name
        }
