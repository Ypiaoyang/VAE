"""
聚类分析脚本

对训练得到的潜在表示进行聚类分析,识别空间域
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
import igraph as ig
import leidenalg as la
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("  聚类分析")
print("=" * 80)

# 1. 加载数据
print("\n步骤 1: 加载数据...")
output_dir = Path('outputs/image_guided_gvae')

latent = np.load(output_dir / 'latent_representations.npy')
kl_div = np.load(output_dir / 'kl_divergence.npy')

# 尝试加载真实标签(如果有)
try:
    true_labels = np.load(output_dir / 'true_labels.npy')
    has_true_labels = True
    print(f"  ✓ 加载真实标签: {len(np.unique(true_labels))} 个簇")
except:
    has_true_labels = False
    print(f"  ℹ 无真实标签 (真实数据)")

print(f"  潜在表示形状: {latent.shape}")
print(f"  KL 散度形状: {kl_div.shape}")

# 2. 构建近邻图
print("\n步骤 2: 构建近邻图...")
# 从潜在表示构建k近邻图
n_neighbors = 10  # 设置近邻数量
knn_graph = kneighbors_graph(latent, n_neighbors=n_neighbors, mode='connectivity', include_self=False)
print(f"  ✓ 已构建 {n_neighbors}-近邻图")

# 3. 使用Leiden算法进行聚类
print("\n步骤 3: 使用Leiden算法进行聚类...")
# 将scipy稀疏矩阵转换为igraph格式
edges = knn_graph.nonzero()
vertices = list(range(latent.shape[0]))

# 创建igraph图
G = ig.Graph()
G.add_vertices(vertices)
G.add_edges(list(zip(edges[0], edges[1])))

# 应用Leiden算法
partition = la.find_partition(G, la.ModularityVertexPartition, n_iterations=-1, seed=42)
cluster_labels = np.array(partition.membership)

# 计算聚类数量
best_k = len(np.unique(cluster_labels))
print(f"  ✓ Leiden算法识别出 {best_k} 个簇")

print(f"  聚类结果:")
for i in range(best_k):
    count = np.sum(cluster_labels == i)
    print(f"    簇 {i}: {count} spots ({count/len(cluster_labels)*100:.1f}%)")

# 4. 评估聚类质量
print(f"\n步骤 4: 评估聚类质量...")
silhouette = silhouette_score(latent, cluster_labels)
print(f"  轮廓系数: {silhouette:.4f}")

if has_true_labels:
    ari = adjusted_rand_score(true_labels, cluster_labels)
    nmi = normalized_mutual_info_score(true_labels, cluster_labels)
    print(f"  调整兰德指数 (ARI): {ari:.4f}")
    print(f"  归一化互信息 (NMI): {nmi:.4f}")

# 5. 分析每个簇的 KL 散度
print(f"\n步骤 5: 分析每个簇的 KL 散度...")
for i in range(best_k):
    cluster_mask = cluster_labels == i
    cluster_kl = kl_div[cluster_mask]
    print(f"  簇 {i}: KL 均值={cluster_kl.mean():.4f}, KL 标准差={cluster_kl.std():.4f}")

# 6. 可视化
print(f"\n步骤 6: 生成可视化...")

# PCA 降维
pca = PCA(n_components=2, random_state=42)
latent_2d = pca.fit_transform(latent)

# 创建图表
fig = plt.figure(figsize=(20, 5))

# 6.1 聚类结果 (潜在空间)
ax1 = plt.subplot(1, 3, 1)
scatter = ax1.scatter(latent_2d[:, 0], latent_2d[:, 1], 
                     c=cluster_labels, cmap='tab10', s=20, alpha=0.7)
ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
ax1.set_title(f'聚类结果 (Leiden, {best_k} 个簇)', fontsize=14, fontweight='bold')
plt.colorbar(scatter, ax=ax1, label='簇标签')

# 6.2 KL 散度 vs 聚类
ax2 = plt.subplot(1, 3, 2)
positions = []
kl_by_cluster = []
for i in range(best_k):
    cluster_mask = cluster_labels == i
    cluster_kl = kl_div[cluster_mask]
    positions.append(i)
    kl_by_cluster.append(cluster_kl)

bp = ax2.boxplot(kl_by_cluster, positions=positions, widths=0.6,
                patch_artist=True, showmeans=True)

# 设置颜色
colors = plt.cm.tab10(np.linspace(0, 1, best_k))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax2.set_xlabel('簇标签', fontsize=12)
ax2.set_ylabel('KL 散度', fontsize=12)
ax2.set_title('各簇的 KL 散度分布', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# 6.3 簇大小分布
ax3 = plt.subplot(1, 3, 3)
cluster_sizes = [np.sum(cluster_labels == i) for i in range(best_k)]
ax3.bar(range(best_k), cluster_sizes, color=colors)
ax3.set_xlabel('簇标签', fontsize=12)
ax3.set_ylabel('簇大小', fontsize=12)
ax3.set_title('簇大小分布', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

plt.tight_layout()

# 保存图表
output_file = output_dir / 'clustering_analysis.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"  ✓ 可视化已保存: {output_file}")
plt.close()

# 7. 保存聚类结果
print(f"\n步骤 7: 保存聚类结果...")
np.save(output_dir / 'cluster_labels.npy', cluster_labels)
print(f"  ✓ 聚类标签已保存: {output_dir / 'cluster_labels.npy'}")

# 8. 生成详细报告
print(f"\n步骤 8: 生成聚类报告...")

report_lines = []
report_lines.append("# 聚类分析报告\n")
report_lines.append(f"## 数据集信息\n")
report_lines.append(f"- Spots: {latent.shape[0]}\n")
report_lines.append(f"- 潜在维度: {latent.shape[1]}\n")
report_lines.append(f"\n## 聚类配置\n")
report_lines.append(f"- 算法: Leiden\n")
report_lines.append(f"- 近邻数量: {n_neighbors}\n")
report_lines.append(f"- 聚类数: {best_k}\n")
report_lines.append(f"- 随机种子: 42\n")
report_lines.append(f"\n## 聚类质量评估\n")
report_lines.append(f"- 轮廓系数: {silhouette:.4f}\n")

if has_true_labels:
    report_lines.append(f"- 调整兰德指数 (ARI): {ari:.4f}\n")
    report_lines.append(f"- 归一化互信息 (NMI): {nmi:.4f}\n")

report_lines.append(f"\n## 簇分布\n")
for i in range(best_k):
    count = np.sum(cluster_labels == i)
    cluster_mask = cluster_labels == i
    cluster_kl = kl_div[cluster_mask]
    report_lines.append(f"### 簇 {i}\n")
    report_lines.append(f"- Spots 数量: {count} ({count/len(cluster_labels)*100:.1f}%)\n")
    report_lines.append(f"- KL 散度均值: {cluster_kl.mean():.4f}\n")
    report_lines.append(f"- KL 散度标准差: {cluster_kl.std():.4f}\n")
    report_lines.append(f"- KL 散度范围: [{cluster_kl.min():.4f}, {cluster_kl.max():.4f}]\n")
    report_lines.append(f"\n")

report_file = output_dir / 'clustering_report.md'
with open(report_file, 'w', encoding='utf-8') as f:
    f.writelines(report_lines)
print(f"  ✓ 聚类报告已保存: {report_file}")

# 显示图表
try:
    plt.show()
except:
    print("  (无法显示图表,但已保存)")

print("\n" + "=" * 80)
print("  聚类分析完成!")
print("=" * 80)

print(f"\n【总结】:")
print(f"  • 最佳聚类数: {best_k}")
print(f"  • 轮廓系数: {silhouette:.4f} (越接近 1 越好)")
print(f"  • 识别出 {best_k} 个空间域")
if has_true_labels:
    print(f"  • 与真实标签的一致性 (ARI): {ari:.4f}")

print(f"\n【输出文件】:")
print(f"  • 聚类可视化: {output_file}")
print(f"  • 聚类标签: {output_dir / 'cluster_labels.npy'}")
print(f"  • 详细报告: {report_file}")

print(f"\n【下一步建议】:")
print(f"  • 空间可视化: 在组织切片上显示聚类结果")
print(f"  • 差异表达分析: 识别每个簇的标志基因")
print(f"  • 功能富集分析: 理解每个簇的生物学功能")
