"""
简化版数据加载脚本 - 直接使用 Squidpy 加载小鼠脑数据

这个脚本会:
1. 下载 Squidpy 内置的小鼠脑 Visium 数据
2. 添加随机图像特征 (用于测试)
3. 保存为 h5ad 格式供模型训练使用
"""

import numpy as np
from pathlib import Path

print("=" * 80)
print("  加载 Squidpy 小鼠脑数据集")
print("=" * 80)

# 1. 加载 Squidpy 数据
print("\n步骤 1: 加载数据...")
try:
    import squidpy as sq
    adata = sq.datasets.visium_hne_adata()
    print(f"✓ 成功加载数据")
    print(f"  Spots: {adata.n_obs}")
    print(f"  Genes: {adata.n_vars}")
    print(f"  Spatial coords: {adata.obsm['spatial'].shape}")
except ImportError as e:
    print(f"✗ 错误: {e}")
    print("  请安装: pip install squidpy scikit-misc")
    exit(1)

# 2. 简化预处理
print("\n步骤 2: 预处理...")

# 过滤低表达基因 (简单版本)
print("  过滤低表达基因...")
from scipy.sparse import issparse
if issparse(adata.X):
    gene_counts = np.array((adata.X > 0).sum(axis=0)).flatten()
else:
    gene_counts = (adata.X > 0).sum(axis=0)

keep_genes = gene_counts >= 10
adata = adata[:, keep_genes].copy()
print(f"  保留 {adata.n_vars} 个基因")

# 选择高变基因 (简单版本 - 按方差)
print("  选择高变基因...")
if issparse(adata.X):
    gene_var = np.array(adata.X.toarray().var(axis=0))
else:
    gene_var = adata.X.var(axis=0)

n_top_genes = min(2000, adata.n_vars)
top_genes_idx = np.argsort(gene_var)[-n_top_genes:]
adata = adata[:, top_genes_idx].copy()
print(f"  选择了 {adata.n_vars} 个高变基因")

# 3. 添加图像特征
print("\n步骤 3: 添加图像特征...")
print("  使用随机特征 (1024 维)")
print("  注意: 这是用于测试,真实应用应从 H&E 图像提取特征")
adata.obsm['image_features'] = np.random.randn(adata.n_obs, 1024).astype(np.float32)
print(f"  图像特征形状: {adata.obsm['image_features'].shape}")

# 4. 保存数据
print("\n步骤 4: 保存数据...")
output_dir = Path('data')
output_dir.mkdir(exist_ok=True)
output_path = output_dir / 'mouse_brain_squidpy.h5ad'

adata.write(output_path)
print(f"✓ 数据已保存到: {output_path}")

# 5. 显示摘要
print("\n" + "=" * 80)
print("  数据准备完成!")
print("=" * 80)
print(f"\n最终数据集:")
print(f"  Spots: {adata.n_obs}")
print(f"  Genes: {adata.n_vars}")
print(f"  Spatial coords: {adata.obsm['spatial'].shape}")
print(f"  Image features: {adata.obsm['image_features'].shape}")

print(f"\n下一步:")
print(f"  运行训练: python main.py --mode train_real --data_path {output_path}")
print(f"  或使用 GPU: python main.py --mode train_real --data_path {output_path} --device cuda")
