"""
加载真实空间转录组学数据的示例脚本

提供三种方法:
1. 使用 Squidpy 内置数据集 (最简单)
2. 从 10x Genomics 下载的数据
3. 从 GEO 或其他来源的 h5ad 文件

运行前需要安装:
pip install squidpy scikit-misc
(可选) pip install scanpy  # 用于更好的预处理
"""

import numpy as np
from pathlib import Path

# scanpy 是可选的,如果没有会使用简化版本
try:
    import scanpy as sc
    HAS_SCANPY = True
except ImportError:
    HAS_SCANPY = False
    print("警告: scanpy 未安装,将使用简化的预处理方法")


def load_squidpy_dataset(dataset_name='visium_hne'):
    """
    方法 1: 使用 Squidpy 内置数据集 (推荐用于快速测试)
    
    Args:
        dataset_name (str): 数据集名称
            - 'visium_hne': 小鼠脑 Visium (H&E 染色)
            - 'visium_fluo': 小鼠肠道 Visium (荧光)
            - 'seqfish': 小鼠胚胎 seqFISH
    
    Returns:
        adata: AnnData object
    """
    try:
        import squidpy as sq
    except ImportError:
        raise ImportError("请安装 squidpy: pip install squidpy")
    
    print(f"加载 Squidpy 内置数据集: {dataset_name}")
    
    if dataset_name == 'visium_hne':
        adata = sq.datasets.visium_hne_adata()
    elif dataset_name == 'visium_fluo':
        adata = sq.datasets.visium_fluo_adata()
    elif dataset_name == 'seqfish':
        adata = sq.datasets.seqfish()
    else:
        raise ValueError(f"未知数据集: {dataset_name}")
    
    print(f"\n数据集信息:")
    print(f"  Spots: {adata.n_obs}")
    print(f"  Genes: {adata.n_vars}")
    print(f"  Spatial coords: {adata.obsm['spatial'].shape}")
    
    return adata


def load_10x_visium(data_path):
    """
    方法 2: 从 10x Genomics 下载的 Visium 数据
    
    Args:
        data_path (str): Visium 数据目录路径
            应包含: filtered_feature_bc_matrix.h5 和 spatial/ 文件夹
    
    Returns:
        adata: AnnData object
    """
    if not HAS_SCANPY:
        raise ImportError("加载 10x Visium 数据需要 scanpy,请安装: pip install scanpy")
    
    print(f"加载 10x Visium 数据: {data_path}")
    
    # 使用 Scanpy 读取 Visium 数据
    adata = sc.read_visium(data_path)
    
    print(f"\n数据集信息:")
    print(f"  Spots: {adata.n_obs}")
    print(f"  Genes: {adata.n_vars}")
    print(f"  Spatial coords: {adata.obsm['spatial'].shape}")
    
    # 检查是否有图像数据
    if 'spatial' in adata.uns:
        print(f"  包含空间图像信息: {list(adata.uns['spatial'].keys())}")
    
    return adata


def load_h5ad_file(file_path):
    """
    方法 3: 从 h5ad 文件加载 (GEO, HTAN 等)
    
    Args:
        file_path (str): h5ad 文件路径
    
    Returns:
        adata: AnnData object
    """
    if not HAS_SCANPY:
        # 如果没有 scanpy,尝试直接用 anndata
        try:
            import anndata
            print(f"加载 h5ad 文件: {file_path}")
            adata = anndata.read_h5ad(file_path)
        except ImportError:
            raise ImportError("需要 scanpy 或 anndata,请安装: pip install scanpy")
    else:
        print(f"加载 h5ad 文件: {file_path}")
        adata = sc.read_h5ad(file_path)
    
    print(f"\n数据集信息:")
    print(f"  Spots: {adata.n_obs}")
    print(f"  Genes: {adata.n_vars}")
    
    # 检查空间坐标
    if 'spatial' in adata.obsm:
        print(f"  Spatial coords: {adata.obsm['spatial'].shape}")
    else:
        print("  警告: 未找到空间坐标 (adata.obsm['spatial'])")
    
    return adata


def preprocess_for_model(adata, n_top_genes=2000, add_image_features=True):
    """
    预处理数据以适配模型
    
    Args:
        adata: AnnData object
        n_top_genes (int): 选择的高变基因数量
        add_image_features (bool): 是否添加图像特征 (如果不存在)
    
    Returns:
        adata: 预处理后的 AnnData object
    """
    print("\n开始预处理...")
    
    # 1. 基因过滤
    print(f"  过滤低表达基因 (min_cells=10)...")
    try:
        import scanpy as sc
        sc.pp.filter_genes(adata, min_cells=10)
        
        # 2. 选择高变基因
        print(f"  选择 {n_top_genes} 个高变基因...")
        # 使用简单的方差方法,不依赖 seurat_v3
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor='seurat')
        
        # 保存原始数据
        adata.layers['counts'] = adata.X.copy()
    except ImportError:
        print(f"  警告: scanpy 未安装,跳过基因过滤和高变基因选择")
        print(f"  如需完整预处理,请安装: pip install scanpy")
        # 简单选择前 n_top_genes 个基因
        if adata.n_vars > n_top_genes:
            adata = adata[:, :n_top_genes].copy()
    
    # 3. 检查图像特征
    if 'image_features' not in adata.obsm:
        if add_image_features:
            print(f"  警告: 未找到图像特征,使用随机特征代替")
            print(f"  建议: 从 H&E 图像提取真实特征以获得更好效果")
            # 使用随机特征 (1024 维,与 ResNet 输出维度一致)
            adata.obsm['image_features'] = np.random.randn(adata.n_obs, 1024).astype(np.float32)
        else:
            raise ValueError("数据集缺少图像特征,请设置 add_image_features=True 或手动添加")
    else:
        print(f"  找到图像特征: {adata.obsm['image_features'].shape}")
    
    # 4. 检查空间坐标
    if 'spatial' not in adata.obsm:
        raise ValueError("数据集缺少空间坐标 (adata.obsm['spatial'])")
    
    print(f"\n预处理完成!")
    print(f"  最终 Spots: {adata.n_obs}")
    print(f"  最终 Genes: {adata.n_vars}")
    print(f"  图像特征维度: {adata.obsm['image_features'].shape[1]}")
    
    return adata


def save_processed_data(adata, output_path):
    """
    保存预处理后的数据
    
    Args:
        adata: AnnData object
        output_path (str): 输出文件路径
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n保存数据到: {output_path}")
    adata.write(output_path)
    print("保存完成!")


def main():
    """主函数: 演示三种加载方法"""
    
    print("=" * 80)
    print("  空间转录组学数据加载示例")
    print("=" * 80)
    
    # ========================================
    # 示例 1: 使用 Squidpy 内置数据 (推荐用于测试)
    # ========================================
    print("\n【示例 1】使用 Squidpy 内置数据集")
    print("-" * 80)
    
    try:
        adata = load_squidpy_dataset('visium_hne')
        adata = preprocess_for_model(adata, n_top_genes=2000)
        save_processed_data(adata, 'data/mouse_brain_squidpy.h5ad')
        
        print("\n✓ 成功! 现在可以运行:")
        print("  python main.py --mode train_real --data_path data/mouse_brain_squidpy.h5ad")
        
    except ImportError as e:
        print(f"\n✗ 失败: {e}")
        print("  请先安装: pip install squidpy")
    
    # ========================================
    # 示例 2: 从 10x Genomics 数据
    # ========================================
    print("\n\n【示例 2】从 10x Genomics Visium 数据")
    print("-" * 80)
    print("如果您已经下载了 10x Genomics 数据,可以使用:")
    print("""
    # 下载示例 (需要先手动下载):
    # wget https://cf.10xgenomics.com/samples/spatial-exp/1.1.0/V1_Adult_Mouse_Brain/V1_Adult_Mouse_Brain_filtered_feature_bc_matrix.h5
    # wget https://cf.10xgenomics.com/samples/spatial-exp/1.1.0/V1_Adult_Mouse_Brain/V1_Adult_Mouse_Brain_spatial.tar.gz
    # tar -xzf V1_Adult_Mouse_Brain_spatial.tar.gz
    
    # 加载数据:
    adata = load_10x_visium('path/to/V1_Adult_Mouse_Brain/')
    adata = preprocess_for_model(adata, n_top_genes=2000)
    save_processed_data(adata, 'data/mouse_brain_10x.h5ad')
    """)
    
    # ========================================
    # 示例 3: 从 h5ad 文件
    # ========================================
    print("\n\n【示例 3】从 h5ad 文件 (GEO, HTAN 等)")
    print("-" * 80)
    print("如果您有 h5ad 文件,可以使用:")
    print("""
    # 加载数据:
    adata = load_h5ad_file('path/to/your_data.h5ad')
    adata = preprocess_for_model(adata, n_top_genes=2000)
    save_processed_data(adata, 'data/processed_data.h5ad')
    """)
    
    print("\n" + "=" * 80)
    print("  完成!")
    print("=" * 80)


if __name__ == '__main__':
    main()
