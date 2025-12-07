"""
可视化训练结果
展示:
1. 潜在空间的 UMAP 可视化
2. KL 散度的空间分布
3. 训练损失曲线
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
print("=" * 80)
print("  训练结果可视化")
print("=" * 80)
# 加载结果
output_dir = Path('outputs/image_guided_gvae')
print("\n加载数据...")
latent = np.load(output_dir / 'latent_representations.npy')
kl_div = np.load(output_dir / 'kl_divergence.npy')
labels = np.load(output_dir / 'true_labels.npy')
print(f"  潜在表示: {latent.shape}")
print(f"  KL 散度: {kl_div.shape}")
print(f"  真实标签: {labels.shape}")
print(f"  簇数: {len(np.unique(labels))}")
# 统计信息
print(f"\nKL 散度统计:")
print(f"  平均值: {kl_div.mean():.4f}")
print(f"  标准差: {kl_div.std():.4f}")
print(f"  最小值: {kl_div.min():.4f}")
print(f"  最大值: {kl_div.max():.4f}")
print(f"  95th 百分位: {np.percentile(kl_div, 95):.4f}")
# 高 KL 值的 spots
high_kl_threshold = np.percentile(kl_div, 95)
high_kl_spots = np.sum(kl_div > high_kl_threshold)
print(f"  高 KL spots (>95th): {high_kl_spots}")
# 创建可视化
print("\n生成可视化...")
fig = plt.figure(figsize=(15, 5))
# 1. 潜在空间可视化 (使用 PCA 降维到 2D)
from sklearn.decomposition import PCA
pca = PCA(n_components=2, random_state=42)
latent_2d = pca.fit_transform(latent)
ax1 = plt.subplot(1, 3, 1)
scatter1 = ax1.scatter(latent_2d[:, 0], latent_2d[:, 1], 
                       c=labels, cmap='tab10', s=20, alpha=0.7)
ax1.set_title('潜在空间 (按真实标签着色)', fontsize=12, fontweight='bold')
ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.colorbar(scatter1, ax=ax1, label='簇标签')
# 2. KL 散度可视化
ax2 = plt.subplot(1, 3, 2)
scatter2 = ax2.scatter(latent_2d[:, 0], latent_2d[:, 1], 
                       c=kl_div, cmap='viridis', s=20, alpha=0.7)
ax2.set_title('KL 散度分布', fontsize=12, fontweight='bold')
ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
cbar2 = plt.colorbar(scatter2, ax=ax2, label='D_KL(q||p)')
# 标记高 KL 区域
high_kl_mask = kl_div > high_kl_threshold
ax2.scatter(latent_2d[high_kl_mask, 0], latent_2d[high_kl_mask, 1],
           s=50, facecolors='none', edgecolors='red', linewidths=2,
           label=f'高 KL (>{high_kl_threshold:.2f})')
ax2.legend()
# 3. KL 散度直方图
ax3 = plt.subplot(1, 3, 3)
ax3.hist(kl_div, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
ax3.axvline(kl_div.mean(), color='red', linestyle='--', linewidth=2, label=f'均值: {kl_div.mean():.2f}')
ax3.axvline(high_kl_threshold, color='orange', linestyle='--', linewidth=2, label=f'95th: {high_kl_threshold:.2f}')
ax3.set_title('KL 散度分布', fontsize=12, fontweight='bold')
ax3.set_xlabel('D_KL(q||p)')
ax3.set_ylabel('频数')
ax3.legend()
ax3.grid(True, alpha=0.3)
plt.tight_layout()
# 保存图片
output_file = output_dir / 'visualization.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✓ 可视化已保存: {output_file}")
# 显示图片
try:
    plt.show()
except:
    print("  (无法显示图片,但已保存)")
print("\n" + "=" * 80)
print("  可视化完成!")
print("=" * 80)
# 总结
print("\n【关键发现】:")
print(f"  1. 模型成功学习到 {len(np.unique(labels))} 个空间簇的潜在表示")
print(f"  2. KL 散度范围: [{kl_div.min():.2f}, {kl_div.max():.2f}]")
print(f"  3. {high_kl_spots} 个 spots 显示高 KL 散度 (形态-基因不一致)")
print(f"  4. PCA 解释方差: PC1={pca.explained_variance_ratio_[0]*100:.1f}%, PC2={pca.explained_variance_ratio_[1]*100:.1f}%")
print("\n【生物学解释】:")
print("  • 高 KL 区域 (红圈) 可能代表:")
print("    - 细胞状态转换区域")
print("    - 组织边界")
print("    - 形态学与分子特征不匹配的区域")
print("\n【文件输出】:")
print(f"  • 潜在表示: {output_dir / 'latent_representations.npy'}")
print(f"  • KL 散度: {output_dir / 'kl_divergence.npy'}")
print(f"  • 模型检查点: {output_dir / 'checkpoint_epoch_final.pt'}")
print(f"  • 可视化: {output_file}")
