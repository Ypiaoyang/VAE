"""
Training Infrastructure

Training loop with KL annealing, gradient clipping, and comprehensive logging.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
import time
import numpy as np
from models.layers import ZINBLoss, GaussianKLDivergence, InfoNCELoss
from models.dec import DEC_Module


class Trainer:
    """
    Trainer for Image-Guided Graph VAE with Deep Embedded Clustering (DEC) support
    
    Implements:
    - Two-stage training: VAE pre-training + DEC fine-tuning
    - KL annealing to prevent posterior collapse
    - Gradient clipping for stability
    - Separate tracking of reconstruction and KL losses
    - Model checkpointing
    
    Args:
        model: ImageGuidedGVAE model
        config: Config object with training parameters
        device: torch device
    """
    
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Loss functions
        self.zinb_loss = ZINBLoss().to(device)
        self.kl_divergence = GaussianKLDivergence().to(device)
        
        # InfoNCE Loss for Contrastive Learning (if enabled)
        self.use_infonce = getattr(self.config.training, 'use_infonce', False)
        if self.use_infonce:
            self.infonce_loss = InfoNCELoss(
                temperature=getattr(self.config.training, 'infonce_temperature', 0.1),
                similarity=getattr(self.config.training, 'infonce_similarity', 'dot')
            ).to(device)
        
        # Optimizer for VAE pre-training
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        # Learning rate scheduler (optional)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=10,
            verbose=True
        )
        
        # DEC related attributes
        self.dec_module = None
        self.dec_optimizer = None
        self.dec_losses = []
        
        # Training state
        self.current_epoch = 0
        self.train_losses = []
        self.val_losses = []
        
        # Create output directory
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def compute_kl_weight(self, epoch):
        """
        Compute KL annealing weight (beta)
        
        Linear annealing from 0 to kl_weight_max over specified epochs
        
        Args:
            epoch (int): Current epoch
        
        Returns:
            float: KL weight
        """
        start = self.config.training.kl_annealing_start
        end = self.config.training.kl_annealing_end
        max_weight = self.config.training.kl_weight_max
        
        if epoch < start:
            return 0.0
        elif epoch >= end:
            return max_weight
        else:
            # Linear annealing
            progress = (epoch - start) / (end - start)
            return progress * max_weight
    
    def train_step(self, data):
        """
        Single training step
        
        Args:
            data: PyTorch Geometric Data object
        
        Returns:
            dict: Loss components
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move data to device
        data = data.to(self.device)
        
        # Forward pass
        output = self.model(data.x, data.img_feat, data.edge_index)
        
        # 1. Reconstruction Loss (ZINB)
        recon_loss = self.zinb_loss(
            x=data.x_counts,
            mean=output['mean'],
            disp=output['disp'],
            pi=output['pi'],
            scale_factor=data.scale_factor
        )
        
        # 2. KL Divergence Loss (Conditional KL)
        mu_q, logvar_q = output['q_dist']
        mu_p, logvar_p = output['p_dist']
        
        kl_loss = self.kl_divergence(mu_q, logvar_q, mu_p, logvar_p)
        
        # Normalize KL by number of spots
        kl_loss = kl_loss / data.x.size(0)
        
        # 3. Total Loss with KL annealing
        kl_weight = self.compute_kl_weight(self.current_epoch)
        total_loss = recon_loss + kl_weight * kl_loss
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            self.config.training.gradient_clip
        )
        
        # Optimizer step
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item(),
            'kl_weight': kl_weight
        }
    
    @torch.no_grad()
    def val_step(self, data):
        """
        Validation step
        
        Args:
            data: PyTorch Geometric Data object
        
        Returns:
            dict: Loss components
        """
        self.model.eval()
        
        # Move data to device
        data = data.to(self.device)
        
        # Forward pass
        output = self.model(data.x, data.img_feat, data.edge_index)
        
        # Compute losses
        recon_loss = self.zinb_loss(
            x=data.x_counts,
            mean=output['mean'],
            disp=output['disp'],
            pi=output['pi'],
            scale_factor=data.scale_factor
        )
        
        mu_q, logvar_q = output['q_dist']
        mu_p, logvar_p = output['p_dist']
        kl_loss = self.kl_divergence(mu_q, logvar_q, mu_p, logvar_p)
        kl_loss = kl_loss / data.x.size(0)
        
        kl_weight = self.compute_kl_weight(self.current_epoch)
        total_loss = recon_loss + kl_weight * kl_loss
        
        return {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item(),
            'kl_weight': kl_weight
        }
    
    def train_epoch(self, train_data):
        """
        Train for one epoch
        
        Args:
            train_data: Training dataset
        
        Returns:
            dict: Average losses for the epoch
        """
        losses = self.train_step(train_data)
        return losses
    
    def validate(self, val_data):
        """
        Validate on validation set
        
        Args:
            val_data: Validation dataset
        
        Returns:
            dict: Validation losses
        """
        losses = self.val_step(val_data)
        return losses
    
    def init_dec_module(self, n_clusters, train_data):
        """
        Initialize DEC module with KMeans on pre-trained latent representations
        
        Args:
            n_clusters (int): Number of clusters
            train_data: Training Data object to extract latent representations
            
        Returns:
            numpy.ndarray: Initial cluster labels from KMeans
        """
        # Create DEC module
        self.dec_module = DEC_Module(
            n_clusters=n_clusters,
            latent_dim=self.model.latent_dim
        ).to(self.device)
        
        # Extract latent representations from pre-trained VAE
        self.model.eval()
        with torch.no_grad():
            data = train_data.to(self.device)
            output = self.model(data.x, data.img_feat, data.edge_index)
            z = output['z'].cpu().numpy()
        
        # Initialize cluster centers
        initial_labels = self.dec_module.init_centers(z)
        
        return initial_labels
    
    def update_target_distribution(self, train_data):
        """
        Update the target distribution P for DEC clustering.
        This should be called at the start of each epoch to avoid moving target problem.
        """
        self.model.eval()
        with torch.no_grad():
            # Get latent representations for all training data
            data = train_data.to(self.device)
            output = self.model(data.x, data.img_feat, data.edge_index)
            z = output['z']
            
            # Calculate current soft assignments Q
            q = self.dec_module(z)
            
            # Update target distribution P based on Q
            self.target_p = self.dec_module.target_distribution(q).detach()
        self.model.train()
        
    def train_step_dec(self, data, gamma=0.1):
        """
        Single training step with DEC loss
        
        Args:
            data: PyTorch Geometric Data object
            gamma (float): Weight for DEC loss
            
        Returns:
            dict: Loss components including DEC loss
        """
        self.model.train()
        self.dec_module.train()
        self.dec_optimizer.zero_grad()
        
        # Move data to device
        data = data.to(self.device)
        
        # Forward pass through VAE
        output = self.model(data.x, data.img_feat, data.edge_index)
        z = output['z']
        
        # 1. Reconstruction Loss (ZINB)
        recon_loss = self.zinb_loss(
            x=data.x_counts,
            mean=output['mean'],
            disp=output['disp'],
            pi=output['pi'],
            scale_factor=data.scale_factor
        )
        
        # 2. KL Divergence Loss (Conditional KL)
        mu_q, logvar_q = output['q_dist']
        mu_p, logvar_p = output['p_dist']
        
        kl_loss = self.kl_divergence(mu_q, logvar_q, mu_p, logvar_p)
        kl_loss = kl_loss / data.x.size(0)
        
        # 3. DEC Loss (KL Divergence between P and Q)
        q = self.dec_module(z)
        dec_loss = F.kl_div(q.log(), self.target_p, reduction='batchmean')
        
        # 4. InfoNCE Loss (Contrastive Learning)
        infonce_loss = 0.0
        if self.use_infonce and hasattr(self, 'infonce_loss'):
            infonce_loss = self.infonce_loss(
                z=z,
                edge_index=data.edge_index
            )
            infonce_weight = getattr(self.config.training, 'infonce_weight', 0.01)
        else:
            infonce_weight = 0.0
        
        # 5. Total Loss - Use max KL weight in DEC phase
        kl_weight = self.config.training.kl_weight_max  # Directly use max KL weight in DEC phase
        total_loss = recon_loss + kl_weight * kl_loss + gamma * dec_loss + infonce_weight * infonce_loss
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            list(self.model.parameters()) + list(self.dec_module.parameters()),
            self.config.training.gradient_clip
        )
        
        # Optimizer step
        self.dec_optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item(),
            'dec_loss': dec_loss.item(),
            'infonce_loss': infonce_loss.item(),
            'kl_weight': kl_weight
        }
    
    def train_with_dec(self, train_data, val_data=None, n_clusters=16, dec_epochs=100, gamma=0.01):
        """
        Two-stage training process:
        1. Pre-train VAE (original training)
        2. Fine-tune with DEC loss
        
        Args:
            train_data: Training Data object
            val_data: Validation Data object (optional)
            n_clusters (int): Number of clusters for DEC
            dec_epochs (int): Number of epochs for DEC fine-tuning
            gamma (float): Weight for DEC loss
        """
        # Stage 1: VAE Pre-training
        print("=" * 80)
        print("Stage 1: VAE Pre-training")
        print("=" * 80)
        
        self.train(train_data, val_data)
        
        # Stage 2: DEC Fine-tuning
        print("\n" + "=" * 80)
        print("Stage 2: DEC Fine-tuning")
        print("=" * 80)
        
        # Initialize DEC module
        initial_labels = self.init_dec_module(n_clusters, train_data)
        
        # Get DEC learning rate (default to 1/10 of pre-training LR)
        dec_lr = getattr(self.config.training, 'dec_learning_rate', None)
        if dec_lr is None:
            dec_lr = self.config.training.learning_rate * 0.1  # 1/10 of pre-training LR
            print(f"  Using DEC learning rate: {dec_lr} (1/10 of pre-training LR)")
        else:
            print(f"  Using DEC learning rate: {dec_lr}")
        
        # Optimizer for joint training (VAE + DEC)
        self.dec_optimizer = optim.Adam(
            list(self.model.parameters()) + list(self.dec_module.parameters()),
            lr=dec_lr,
            weight_decay=self.config.training.weight_decay
        )
        
        # Initialize target distribution P
        with torch.no_grad():
            data = train_data.to(self.device)
            output = self.model(data.x, data.img_feat, data.edge_index)
            z = output['z']
            q = self.dec_module(z)
            self.target_p = self.dec_module.target_distribution(q).detach()
        
        print(f"Starting DEC fine-tuning for {dec_epochs} epochs")
        print(f"DEC loss weight (gamma): {gamma}")
        print(f"Number of clusters: {n_clusters}")
        print("-" * 80)
        
        # Get target distribution update frequency
        target_update_freq = getattr(self.config.training, 'target_distribution_update_freq', 1)
        
        # DEC training loop
        for epoch in range(dec_epochs):
            self.current_epoch = epoch
            start_time = time.time()
            
            # Update target distribution P every N epochs (or always on first epoch)
            if epoch == 0 or epoch % target_update_freq == 0:
                self.update_target_distribution(train_data)
            
            # Training with DEC
            train_losses = self.train_step_dec(train_data, gamma=gamma)
            self.train_losses.append(train_losses)
            
            epoch_time = time.time() - start_time
            
            # Logging
            if epoch % self.config.training.log_interval == 0:
                log_str = f"DEC Epoch {epoch:4d} | Time: {epoch_time:.2f}s | "
                log_str += f"Total Loss: {train_losses['total_loss']:.4f} "
                log_str += f"(Recon: {train_losses['recon_loss']:.4f}, "
                log_str += f"KL: {train_losses['kl_loss']:.4f}, "
                if 'infonce_loss' in train_losses and train_losses['infonce_loss'] > 0:
                    log_str += f"InfoNCE: {train_losses['infonce_loss']:.4f}, "
                log_str += f"β: {train_losses['kl_weight']:.3f})"
                print(log_str)
            
            # Save checkpoint during DEC training
            if epoch % self.config.training.save_interval == 0:
                self.save_checkpoint(f'dec_{epoch}')
        
        # Save final DEC model
        self.save_checkpoint('dec_final')
        print("\nDEC fine-tuning completed!")
        
        # Print InfoNCE configuration if enabled
        if self.use_infonce:
            print("\nContrastive Learning (InfoNCE) Configuration:")
            print(f"- Enabled: {self.use_infonce}")
            print(f"- Weight: {getattr(self.config.training, 'infonce_weight', 0.01)}")
            print(f"- Temperature: {getattr(self.config.training, 'infonce_temperature', 0.1)}")
            print(f"- Similarity: {getattr(self.config.training, 'infonce_similarity', 'dot')}")
    
    def train(self, train_data, val_data=None):
        """
        Original VAE training loop (pre-training stage)
        
        Args:
            train_data: Training Data object
            val_data: Validation Data object (optional)
        """
        print(f"Starting VAE pre-training for {self.config.training.epochs} epochs")
        print(f"Output directory: {self.output_dir}")
        print(f"Device: {self.device}")
        print("-" * 80)
        
        for epoch in range(self.config.training.epochs):
            self.current_epoch = epoch
            start_time = time.time()
            
            # Training
            train_losses = self.train_epoch(train_data)
            self.train_losses.append(train_losses)
            
            # Validation
            if val_data is not None:
                val_losses = self.validate(val_data)
                self.val_losses.append(val_losses)
            else:
                val_losses = None
            
            epoch_time = time.time() - start_time
            
            # Logging
            if epoch % self.config.training.log_interval == 0:
                self._log_epoch(epoch, train_losses, val_losses, epoch_time)
            
            # Learning rate scheduling
            if val_data is not None:
                self.scheduler.step(val_losses['total_loss'])
            
            # Save checkpoint
            if epoch % self.config.training.save_interval == 0:
                self.save_checkpoint(epoch)
        
        # Save final pre-trained model
        self.save_checkpoint('pre_final')
        print("\nVAE pre-training completed!")
    
    def _log_epoch(self, epoch, train_losses, val_losses, epoch_time):
        """Log epoch statistics"""
        log_str = f"Epoch {epoch:4d} | Time: {epoch_time:.2f}s | "
        log_str += f"Train Loss: {train_losses['total_loss']:.4f} "
        log_str += f"(Recon: {train_losses['recon_loss']:.4f}, "
        log_str += f"KL: {train_losses['kl_loss']:.4f}, "
        log_str += f"β: {train_losses['kl_weight']:.3f})"
        
        if val_losses is not None:
            log_str += f" | Val Loss: {val_losses['total_loss']:.4f}"
        
        print(log_str)
    
    def save_checkpoint(self, epoch):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config.to_dict()
        }
        
        path = self.output_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        print(f"Loaded checkpoint from {path} (epoch {self.current_epoch})")
