"""
Spatial Transcriptomics Dataset

Data loading and preprocessing for spatial transcriptomics with image features.
Supports integration with AnnData/Scanpy ecosystem.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from scipy.spatial import Delaunay
from sklearn.neighbors import NearestNeighbors
import warnings


class SpatialTranscriptomicsDataset(Dataset):
    """
    Dataset for spatial transcriptomics with image features
    
    This class handles:
    - Gene expression normalization
    - Spatial graph construction
    - Image feature extraction/loading
    - PyTorch Geometric Data object creation
    
    Args:
        gene_expression (np.ndarray): Count matrix [N_spots, N_genes]
        spatial_coords (np.ndarray): Spatial coordinates [N_spots, 2]
        image_features (np.ndarray): Pre-extracted image features [N_spots, img_dim]
        graph_type (str): "knn" or "radius" for graph construction
        k_neighbors (int): Number of neighbors for k-NN graph
        radius (float): Radius for radius graph
        normalize (bool): Apply total count normalization
        log_transform (bool): Apply log(1+x) transformation
    """
    
    def __init__(
        self,
        gene_expression,
        spatial_coords,
        image_features=None,
        graph_type="knn",
        k_neighbors=6,
        radius=None,
        normalize=True,
        log_transform=True
    ):
        super().__init__()
        
        self.gene_expression = gene_expression
        self.spatial_coords = spatial_coords
        self.image_features = image_features
        self.graph_type = graph_type
        self.k_neighbors = k_neighbors
        self.radius = radius
        
        # Validate inputs
        assert gene_expression.shape[0] == spatial_coords.shape[0], \
            "Number of spots must match between gene expression and coordinates"
        
        if image_features is not None:
            assert image_features.shape[0] == gene_expression.shape[0], \
                "Number of spots must match for image features"
        
        # Preprocess gene expression
        self.gene_expression_processed = self._preprocess_gene_expression(
            gene_expression, normalize, log_transform
        )
        
        # Store library size factors (for ZINB loss)
        self.library_size = np.sum(gene_expression, axis=1, keepdims=True)
        self.library_size_factors = self.library_size / np.mean(self.library_size)
        
        # Construct spatial graph
        self.edge_index = self._construct_spatial_graph()
        
        # Handle image features
        if image_features is None:
            warnings.warn(
                "No image features provided. Using zero vectors. "
                "For real applications, provide pre-extracted image embeddings."
            )
            self.image_features = np.zeros((gene_expression.shape[0], 1024))
        
        # Convert to tensors
        self.x = torch.FloatTensor(self.gene_expression_processed)
        self.img = torch.FloatTensor(self.image_features)
        self.pos = torch.FloatTensor(self.spatial_coords)
        self.edge_index = torch.LongTensor(self.edge_index)
        self.scale_factors = torch.FloatTensor(self.library_size_factors)
        
        # Original counts (for loss computation)
        self.x_counts = torch.FloatTensor(gene_expression)
    
    def _preprocess_gene_expression(self, counts, normalize, log_transform):
        """
        Preprocess gene expression counts
        
        Args:
            counts (np.ndarray): Raw count matrix
            normalize (bool): Apply total count normalization
            log_transform (bool): Apply log transformation
        
        Returns:
            np.ndarray: Processed gene expression
        """
        processed = counts.copy()
        
        # Total count normalization (library size normalization)
        if normalize:
            library_size = np.sum(processed, axis=1, keepdims=True)
            processed = processed / (library_size + 1e-10) * 1e4  # Scale to 10,000
        
        # Log transformation
        if log_transform:
            processed = np.log1p(processed)
        
        return processed
    
    def _construct_spatial_graph(self):
        """
        Construct spatial graph based on coordinates
        
        Returns:
            np.ndarray: Edge index [2, E]
        """
        if self.graph_type == "knn":
            return self._knn_graph()
        elif self.graph_type == "radius":
            return self._radius_graph()
        elif self.graph_type == "delaunay":
            return self._delaunay_graph()
        else:
            raise ValueError(f"Unknown graph type: {self.graph_type}")
    
    def _knn_graph(self):
        """k-Nearest Neighbors graph"""
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors + 1).fit(self.spatial_coords)
        distances, indices = nbrs.kneighbors(self.spatial_coords)
        
        # Build edge list (exclude self-loops)
        edge_list = []
        for i in range(len(indices)):
            for j in range(1, len(indices[i])):  # Skip first (self)
                edge_list.append([i, indices[i][j]])
                edge_list.append([indices[i][j], i])  # Bidirectional
        
        edge_index = np.array(edge_list).T
        return edge_index
    
    def _radius_graph(self):
        """Radius-based graph"""
        if self.radius is None:
            raise ValueError("Radius must be specified for radius graph")
        
        nbrs = NearestNeighbors(radius=self.radius).fit(self.spatial_coords)
        distances, indices = nbrs.radius_neighbors(self.spatial_coords)
        
        edge_list = []
        for i, neighbors in enumerate(indices):
            for j in neighbors:
                if i != j:  # Exclude self-loops
                    edge_list.append([i, j])
        
        edge_index = np.array(edge_list).T
        return edge_index
    
    def _delaunay_graph(self):
        """Delaunay triangulation graph"""
        tri = Delaunay(self.spatial_coords)
        
        edge_set = set()
        for simplex in tri.simplices:
            # Add all edges of the triangle
            for i in range(3):
                edge = tuple(sorted([simplex[i], simplex[(i+1)%3]]))
                edge_set.add(edge)
        
        # Convert to bidirectional edge list
        edge_list = []
        for edge in edge_set:
            edge_list.append([edge[0], edge[1]])
            edge_list.append([edge[1], edge[0]])
        
        edge_index = np.array(edge_list).T
        return edge_index
    
    def __len__(self):
        return 1  # Entire dataset as single graph
    
    def __getitem__(self, idx):
        """
        Return PyTorch Geometric Data object
        
        Returns:
            Data: Contains x, img_feat, edge_index, pos, scale_factor, x_counts
        """
        return Data(
            x=self.x,                    # Processed gene expression
            img_feat=self.img,           # Image features
            edge_index=self.edge_index,  # Spatial graph
            pos=self.pos,                # Spatial coordinates
            scale_factor=self.scale_factors,  # Library size factors
            x_counts=self.x_counts       # Original counts (for ZINB loss)
        )


def load_from_anndata(adata, config):
    """
    Create dataset from AnnData object
    
    Args:
        adata: AnnData object with spatial transcriptomics data
        config: DataConfig object
    
    Returns:
        SpatialTranscriptomicsDataset
    
    Expected AnnData structure:
        - adata.X: Gene expression matrix
        - adata.obsm['spatial']: Spatial coordinates
        - adata.obsm['image_features']: Pre-extracted image features (optional)
    """
    # Extract data
    gene_expression = adata.X
    if hasattr(gene_expression, 'toarray'):
        gene_expression = gene_expression.toarray()
    
    spatial_coords = adata.obsm['spatial']
    
    # Image features (if available)
    image_features = None
    if 'image_features' in adata.obsm:
        image_features = adata.obsm['image_features']
    
    # Highly variable genes selection
    if config.highly_variable_genes is not None and hasattr(adata, 'var'):
        if 'highly_variable' in adata.var.columns:
            hvg_mask = adata.var['highly_variable'].values
            gene_expression = gene_expression[:, hvg_mask]
    
    # Create dataset
    dataset = SpatialTranscriptomicsDataset(
        gene_expression=gene_expression,
        spatial_coords=spatial_coords,
        image_features=image_features,
        graph_type=config.graph_type,
        k_neighbors=config.k_neighbors,
        radius=config.radius,
        normalize=config.normalize_total,
        log_transform=config.log_transform
    )
    
    return dataset


def create_synthetic_data(n_spots=1000, n_genes=2000, n_clusters=5, seed=42):
    """
    Create synthetic spatial transcriptomics data for testing
    
    Args:
        n_spots (int): Number of spatial spots
        n_genes (int): Number of genes
        n_clusters (int): Number of spatial clusters
        seed (int): Random seed
    
    Returns:
        tuple: (gene_expression, spatial_coords, image_features, labels)
    """
    np.random.seed(seed)
    
    # Generate spatial coordinates in a grid-like pattern
    grid_size = int(np.ceil(np.sqrt(n_spots)))
    x = np.linspace(0, 10, grid_size)
    y = np.linspace(0, 10, grid_size)
    xx, yy = np.meshgrid(x, y)
    spatial_coords_full = np.column_stack([xx.ravel(), yy.ravel()])
    
    # Take exactly n_spots coordinates
    spatial_coords = spatial_coords_full[:n_spots].copy()
    
    # Add noise to coordinates
    spatial_coords += np.random.randn(n_spots, 2) * 0.1
    
    # Generate cluster centers
    cluster_centers = np.random.rand(n_clusters, 2) * 10
    
    # Assign spots to clusters based on distance
    distances = np.linalg.norm(
        spatial_coords[:, np.newaxis, :] - cluster_centers[np.newaxis, :, :], 
        axis=2
    )
    labels = np.argmin(distances, axis=1)
    
    # Generate gene expression with cluster-specific patterns
    gene_expression = np.zeros((n_spots, n_genes))
    for i in range(n_clusters):
        cluster_mask = labels == i
        n_cluster_spots = np.sum(cluster_mask)
        
        # Cluster-specific gene expression pattern
        cluster_mean = np.random.rand(n_genes) * 100 + 20
        gene_expression[cluster_mask] = np.random.poisson(
            cluster_mean[np.newaxis, :], 
            size=(n_cluster_spots, n_genes)
        )
    
    # Generate synthetic image features (correlated with spatial position)
    image_features = np.random.randn(n_spots, 1024)
    for i in range(n_clusters):
        cluster_mask = labels == i
        cluster_feature = np.random.randn(1024)
        image_features[cluster_mask] += cluster_feature[np.newaxis, :]
    
    return gene_expression, spatial_coords, image_features, labels
