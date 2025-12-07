import scanpy as sc
import numpy as np
import pandas as pd

"""
分析mouse_brain_squidpy.h5ad的具体结构
该脚本将详细展示AnnData对象的所有主要组件
"""

def analyze_anndata_structure(adata_path):
    """分析AnnData对象的结构并打印详细信息"""
    print(f"=" * 80)
    print(f"分析 AnnData 文件：{adata_path}")
    print(f"=" * 80)
    
    # 加载数据
    adata = sc.read_h5ad(adata_path)
    
    print(f"\n📊 基本信息：")
    print(f"  - 观测值数量 (spots): {adata.n_obs}")
    print(f"  - 变量数量 (genes): {adata.n_vars}")
    print(f"  - 数据类型: {type(adata.X).__name__}")
    
    # 分析adata.X
    print(f"\n📋 X矩阵信息：")
    print(f"  - 形状: {adata.X.shape}")
    if hasattr(adata.X, 'toarray'):
        print(f"  - 存储格式: 稀疏矩阵 ({type(adata.X).__name__})")
        # 计算稀疏度
        dense_shape = adata.X.shape[0] * adata.X.shape[1]
        non_zero = adata.X.nnz
        print(f"  - 稀疏度: {100 * (1 - non_zero / dense_shape):.4f}%")
    else:
        print(f"  - 存储格式: 密集矩阵")
    
    # 分析adata.obs
    print(f"\n👥 观测值信息 (adata.obs):")
    print(f"  - 列数: {adata.obs.shape[1]}")
    print(f"  - 列名: {list(adata.obs.columns)}")
    if adata.obs.shape[1] > 0:
        print(f"\n  示例数据 (前3行):")
        print(adata.obs.head(3).to_string())
    
    # 分析adata.var
    print(f"\n🔬 变量信息 (adata.var):")
    print(f"  - 列数: {adata.var.shape[1]}")
    print(f"  - 列名: {list(adata.var.columns)}")
    if adata.var.shape[1] > 0:
        print(f"\n  示例数据 (前3行):")
        print(adata.var.head(3).to_string())
    
    # 分析adata.obsm
    print(f"\n📍 观测值多维注释 (adata.obsm):")
    print(f"  - 键数量: {len(adata.obsm.keys())}")
    for key in adata.obsm.keys():
        data = adata.obsm[key]
        print(f"\n    🔑 {key}:")
        print(f"      - 类型: {type(data).__name__}")
        print(f"      - 形状: {data.shape}")
        if hasattr(data, 'dtype'):
            print(f"      - 数据类型: {data.dtype}")
        if isinstance(data, np.ndarray) and data.size > 0:
            print(f"      - 示例值: {data[0][:5] if data.ndim > 1 else data[:5]}")
    
    # 分析adata.varm
    print(f"\n🔬 变量多维注释 (adata.varm):")
    print(f"  - 键数量: {len(adata.varm.keys())}")
    for key in adata.varm.keys():
        data = adata.varm[key]
        print(f"\n    🔑 {key}:")
        print(f"      - 类型: {type(data).__name__}")
        print(f"      - 形状: {data.shape}")
        if hasattr(data, 'dtype'):
            print(f"      - 数据类型: {data.dtype}")
    
    # 分析adata.uns
    print(f"\n📚 非结构化注释 (adata.uns):")
    print(f"  - 键数量: {len(adata.uns.keys())}")
    for key in adata.uns.keys():
        data = adata.uns[key]
        print(f"\n    🔑 {key}:")
        print(f"      - 类型: {type(data).__name__}")
        if isinstance(data, dict):
            print(f"      - 子键: {list(data.keys())}")
        elif hasattr(data, 'shape'):
            print(f"      - 形状: {data.shape}")
    
    # 分析空间信息
    print(f"\n🗺️  空间信息分析：")
    if 'spatial' in adata.obsm:
        spatial_coords = adata.obsm['spatial']
        print(f"  - 空间坐标存在于 adata.obsm['spatial']")
        print(f"  - 坐标范围：")
        print(f"      X轴: [{spatial_coords[:, 0].min():.2f}, {spatial_coords[:, 0].max():.2f}]")
        print(f"      Y轴: [{spatial_coords[:, 1].min():.2f}, {spatial_coords[:, 1].max():.2f}]")
    
    if 'spatial' in adata.uns:
        print(f"  - 空间元数据存在于 adata.uns['spatial']")
        spatial_meta = adata.uns['spatial']
        if isinstance(spatial_meta, dict):
            print(f"      - 包含的键: {list(spatial_meta.keys())}")
    
    # 分析图像特征
    print(f"\n🖼️  图像特征分析：")
    if 'image_features' in adata.obsm:
        img_feat = adata.obsm['image_features']
        print(f"  - 图像特征存在于 adata.obsm['image_features']")
        print(f"  - 特征维度: {img_feat.shape[1]}")
        print(f"  - 特征统计：")
        print(f"      最小值: {img_feat.min():.4f}")
        print(f"      最大值: {img_feat.max():.4f}")
        print(f"      平均值: {img_feat.mean():.4f}")
        print(f"      标准差: {img_feat.std():.4f}")
        
        # 检查是否有零向量
        zero_vectors = np.sum(np.all(img_feat == 0, axis=1))
        print(f"  - 零向量数量: {zero_vectors} ({zero_vectors/img_feat.shape[0]*100:.2f}%)")
    
    print(f"\n" + "=" * 80)
    print("分析完成！")
    print(f"=" * 80)

if __name__ == "__main__":
    analyze_anndata_structure("data/mouse_intestine_visium_fluo.h5ad")