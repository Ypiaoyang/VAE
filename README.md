# Image-Guided Graph VAE for Spatial Transcriptomics

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Conditional Variational Autoencoder (CVAE) for spatial transcriptomics analysis where tissue images serve as **prior conditions** rather than reconstruction targets.

## 🔬 Core Innovation

Unlike traditional approaches that attempt to reconstruct images (e.g., DeepST), this model uses image features to **guide the latent space distribution**:

- **Prior Network**: $p_\theta(Z | I, A)$ - learned from image and spatial graph
- **Posterior Network**: $q_\phi(Z | X, I, A)$ - inferred from gene expression, image, and spatial graph
- **Reconstruction**: Gene expression only (ZINB distribution)

This creates a biologically meaningful framework where **high KL divergence indicates discordance between morphology and molecular state** - precisely where biological discovery happens.

## 📊 Mathematical Foundation

### Zero-Inflated Negative Binomial (ZINB) Loss

Models sparse gene expression counts with zero-inflation:

```
P(X=x | μ, θ, π) = π·δ₀(x) + (1-π)·NB(x; μ, θ)
```

### Conditional KL Divergence

Forces alignment between gene-driven and morphology-driven latent representations:

```
D_KL(q||p) = 0.5 · Σ[(σ²_q + (μ_q - μ_p)²)/σ²_p + log(σ²_p/σ²_q) - 1]
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd newproject

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Training on Synthetic Data

```bash
python main.py --mode train_synthetic --device cuda --seed 42
```

This will:
1. Generate synthetic spatial transcriptomics data with 5 spatial clusters
2. Train the model with KL annealing
3. Extract latent representations and KL divergence patterns
4. Save results to `./outputs/image_guided_gvae/`

### Training on Real Data

```bash
python main.py --mode train_real \
    --data_path path/to/your_data.h5ad \
    --device cuda \
    --output ./outputs
```

**Expected AnnData structure**:
- `adata.X`: Gene expression matrix
- `adata.obsm['spatial']`: Spatial coordinates
- `adata.obsm['image_features']`: Pre-extracted image features (optional)

## 📁 Project Structure

```
newproject/
├── models/
│   ├── layers.py       # ZINB loss and KL divergence
│   └── cvae.py         # Image-Guided GVAE architecture
├── data/
│   └── dataset.py      # Data loading and graph construction
├── utils/
│   └── config.py       # Configuration management
├── trainer.py          # Training loop with KL annealing
├── main.py             # CLI entry point
├── requirements.txt
└── README.md
```

## 🔧 Configuration

Create a `config.json` file:

```json
{
  "model": {
    "hidden_dim": 256,
    "latent_dim": 32,
    "num_heads": 3,
    "dropout": 0.1
  },
  "training": {
    "learning_rate": 0.001,
    "epochs": 200,
    "kl_annealing_start": 0,
    "kl_annealing_end": 50,
    "kl_weight_max": 1.0
  },
  "data": {
    "graph_type": "knn",
    "k_neighbors": 6,
    "highly_variable_genes": 2000
  }
}
```

Then run:
```bash
python main.py --mode train_real --data_path data.h5ad --config config.json
```

## 🧬 Model Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Input Data                               │
│  Gene Expression (X)  │  Image Features (I)  │  Graph (A)   │
└────────┬──────────────┴──────────────┬───────────────┬──────┘
         │                             │               │
         │                             │               │
    ┌────▼────┐                   ┌────▼────┐         │
    │  Gene   │                   │  Image  │         │
    │ Encoder │                   │ Encoder │         │
    └────┬────┘                   └────┬────┘         │
         │                             │               │
         │        ┌────────────────────┘               │
         │        │                                    │
         │        │            ┌───────────────────────┘
         │        │            │
    ┌────▼────────▼──┐    ┌────▼────┐
    │  Posterior (q) │    │ Prior(p)│  ← Only image+graph!
    │  GAT Encoder   │    │   GAT   │
    │  (μ_q, σ_q)    │    │(μ_p,σ_p)│
    └────┬───────────┘    └────┬────┘
         │                     │
         └──────┬──────────────┘
                │  KL Divergence
           ┌────▼────┐
           │ Sampling│ z ~ N(μ_q, σ²_q)
           └────┬────┘
                │
           ┌────▼────┐
           │ Decoder │
           │   MLP   │
           └────┬────┘
                │
        ┌───────┴───────┐
        │  ZINB Params  │
        │  (μ, θ, π)    │
        └───────────────┘
```

## 📈 Key Features

### 1. **ZINB Reconstruction Loss**
- Handles extreme sparsity in spatial transcriptomics
- Numerically stable implementation with log-sum-exp
- Library size normalization

### 2. **KL Annealing**
- Prevents posterior collapse
- Linear annealing from β=0 to β=1
- Configurable annealing schedule

### 3. **Graph Attention Networks**
- Spatial context through GAT layers
- Multi-head attention mechanism
- Choice of k-NN or radius graphs

### 4. **Prior Network Independence**
- Prior network receives **only** image + graph
- Ensures morphology-based guidance
- Enables biological interpretation through KL divergence

## 🔬 Biological Interpretation

### High KL Divergence Regions

Spots with high $D_{KL}(q||p)$ indicate **discordance between morphology and gene expression**:

- Morphologically normal tissue with pre-cancerous gene signatures
- Immune infiltration in homogeneous-looking tissue
- Cellular state transitions not visible in H&E staining

### Latent Space Analysis

The learned latent space $Z$ can be used for:
- Spatial domain identification (clustering)
- Trajectory inference
- Differential expression analysis
- Integration with other modalities

## 📊 Example Usage

```python
from models import ImageGuidedGVAE
from data.dataset import SpatialTranscriptomicsDataset
from utils import Config

# Create config
config = Config()

# Load your data
dataset = SpatialTranscriptomicsDataset(
    gene_expression=gene_expr,
    spatial_coords=coords,
    image_features=img_feat,
    graph_type="knn",
    k_neighbors=6
)

# Create model
model = ImageGuidedGVAE(
    input_dim=2000,
    img_dim=1024,
    hidden_dim=256,
    latent_dim=32
)

# Train
from trainer import Trainer
trainer = Trainer(model, config, device='cuda')
trainer.train(dataset[0])
```

## 🎯 Future Enhancements

- [ ] Contrastive pre-training for image-gene alignment (MuCoST-style)
- [ ] Multi-sample batch training
- [ ] Integration with foundation models (UNI, Virchow)
- [ ] Batch effect correction
- [ ] Interactive visualization tools

## 📚 References

1. **SpaICL**: Multi-modal analysis with contrastive learning
2. **MuCoST**: Contrastive learning for spatial omics
3. **STAGATE**: Graph attention for spatial transcriptomics
4. **DeepST**: Deep learning for spatial transcriptomics (image reconstruction approach)

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{image_guided_gvae,
  title={Image-Guided Graph VAE for Spatial Transcriptomics},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/image-guided-gvae}
}
```

## 📄 License

MIT License - see LICENSE file for details

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

**Note**: This is a research implementation. For production use, additional validation and optimization may be required.
