"""
查看训练结果摘要
"""
import numpy as np
from pathlib import Path
print("=" * 80)
print("  训练结果摘要")
print("=" * 80)
# 加载结果
output_dir = Path('outputs/image_guided_gvae')
print("\n【文件列表】:")
for file in sorted(output_dir.glob('*')):
    size_mb = file.stat().st_size / (1024 * 1024)
    print(f"  • {file.name:30s} ({size_mb:6.2f} MB)")
print("\n【数据摘要】:")
# 1. 潜在表示
latent = np.load(output_dir / 'latent_representations.npy')
print(f"\n1. 潜在表示 (Latent Representations)")
print(f"   形状: {latent.shape}")
print(f"   数据类型: {latent.dtype}")
print(f"   范围: [{latent.min():.4f}, {latent.max():.4f}]")
print(f"   均值: {latent.mean():.4f}")
print(f"   标准差: {latent.std():.4f}")
print(f"\n   前3个spots的潜在表示示例:")
for i in range(min(3, latent.shape[0])):
    print(f"   Spot {i}: {latent[i, :5]} ... (显示前5维)")
# 2. KL 散度
kl_div = np.load(output_dir / 'kl_divergence.npy')
print(f"\n2. KL 散度 (KL Divergence)")
print(f"   形状: {kl_div.shape}")
print(f"   数据类型: {kl_div.dtype}")
print(f"   统计:")
print(f"     • 最小值: {kl_div.min():.4f}")
print(f"     • 最大值: {kl_div.max():.4f}")
print(f"     • 平均值: {kl_div.mean():.4f}")
print(f"     • 中位数: {np.median(kl_div):.4f}")
print(f"     • 标准差: {kl_div.std():.4f}")
print(f"     • 95th 百分位: {np.percentile(kl_div, 95):.4f}")
# 高KL区域
high_kl_threshold = np.percentile(kl_div, 95)
high_kl_spots = np.where(kl_div > high_kl_threshold)[0]
print(f"\n   高 KL 散度 spots (>95th 百分位):")
print(f"     • 数量: {len(high_kl_spots)}")
print(f"     • 索引: {high_kl_spots[:10].tolist()} ..." if len(high_kl_spots) > 10 else f"     • 索引: {high_kl_spots.tolist()}")
# 3. 真实标签
labels = np.load(output_dir / 'true_labels.npy')
print(f"\n3. 真实标签 (True Labels)")
print(f"   形状: {labels.shape}")
print(f"   簇数: {len(np.unique(labels))}")
print(f"   标签分布:")
for label in np.unique(labels):
    count = np.sum(labels == label)
    print(f"     • 簇 {label}: {count} spots ({count/len(labels)*100:.1f}%)")
print("\n" + "=" * 80)
print("  ✓ 训练成功完成!")
print("=" * 80)
print("\n【关键发现】:")
print(f"  • 模型学习到 {latent.shape[1]} 维的潜在表示")
print(f"  • 识别出 {len(high_kl_spots)} 个高 KL 散度区域 (形态-基因不一致)")
print(f"  • KL 散度范围: [{kl_div.min():.2f}, {kl_div.max():.2f}]")
print("\n【生物学意义】:")
print("  • 高 KL 区域可能代表:")
print("    - 细胞状态转换边界")
print("    - 组织微环境变化区域")
print("    - 形态学与分子特征不匹配的spots")
print("\n【下一步分析】:")
print("  • 运行可视化: python visualize_results.py")
print("  • 使用潜在表示进行聚类分析")
print("  • 分析高 KL 区域的基因表达特征")
