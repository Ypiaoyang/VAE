#!/usr/bin/env python3
"""
加载小鼠肠道 Visium 荧光数据集 (visium_fluo) 的脚本

该脚本将：
1. 使用 Squidpy 加载内置的 visium_fluo 数据集
2. 对数据进行预处理（选择高变基因等）
3. 添加图像特征（如果不存在）
4. 保存处理后的数据到 h5ad 文件
"""

import sys
import os

# 添加项目根目录到 Python 路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from load_real_data_example import load_squidpy_dataset, preprocess_for_model, save_processed_data

def main():
    """主函数：加载、预处理和保存 visium_fluo 数据集"""
    
    print("=" * 80)
    print("  加载小鼠肠道 Visium 荧光数据集 (visium_fluo)")
    print("=" * 80)
    
    try:
        # 1. 加载 visium_fluo 数据集
        print("\n1. 正在加载 visium_fluo 数据集...")
        adata = load_squidpy_dataset('visium_fluo')
        
        # 2. 预处理数据
        print("\n2. 正在预处理数据...")
        adata = preprocess_for_model(
            adata, 
            n_top_genes=2000,  # 选择 2000 个高变基因
            add_image_features=True  # 如果没有图像特征则添加
        )
        
        # 3. 保存处理后的数据
        output_file = 'data/mouse_intestine_visium_fluo.h5ad'
        print(f"\n3. 正在保存数据到: {output_file}")
        save_processed_data(adata, output_file)
        
        print("\n" + "=" * 80)
        print("✓ 数据集加载和预处理完成!")
        print("=" * 80)
        print(f"数据集信息:")
        print(f"- 样本数量 (spots): {adata.n_obs}")
        print(f"- 基因数量: {adata.n_vars}")
        print(f"- 空间坐标维度: {adata.obsm['spatial'].shape}")
        print(f"- 图像特征维度: {adata.obsm['image_features'].shape}")
        print(f"- 保存路径: {output_file}")
        print("\n可以使用以下命令训练模型:")
        print(f"python main.py --mode train_real --data_path {output_file}")
        
    except ImportError as e:
        print(f"\n✗ 导入错误: {e}")
        print("请确保已安装必要的依赖:")
        print("pip install squidpy scikit-misc scanpy")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        print("加载数据集时发生错误")
        sys.exit(1)

if __name__ == '__main__':
    main()