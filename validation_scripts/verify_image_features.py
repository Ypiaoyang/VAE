import scanpy as sc
import numpy as np

# 加载真实数据
adata_path = 'data/mouse_brain_squidpy.h5ad'
print(f"Loading data from {adata_path}...")
adata = sc.read_h5ad(adata_path)

# 检查数据基本信息
print(f"Dataset: {adata.shape[0]} spots, {adata.shape[1]} genes")

# 检查obsm中的键
print("\nKeys in adata.obsm:")
for key in adata.obsm.keys():
    print(f"  - {key}")

# 检查是否存在image_features
if 'image_features' in adata.obsm:
    image_features = adata.obsm['image_features']
    print(f"\nImage features found!")
    print(f"  Shape: {image_features.shape}")
    print(f"  Type: {type(image_features)}")
    print(f"  First few values: {image_features[0][:5]}")
    print(f"  Non-zero values: {np.count_nonzero(image_features) / image_features.size:.2%}")
else:
    print(f"\n❌ No image_features found in adata.obsm!")
    print("The model will use zero vectors for image features.")

# 检查空间坐标是否存在
if 'spatial' in adata.obsm:
    spatial_coords = adata.obsm['spatial']
    print(f"\nSpatial coordinates found!")
    print(f"  Shape: {spatial_coords.shape}")
    print(f"  First few coordinates: {spatial_coords[:3]}")
else:
    print(f"\n❌ No spatial coordinates found in adata.obsm!")