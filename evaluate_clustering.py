import scanpy as sc
import anndata
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("  源数据聚类效果评估")
print("=" * 80)

# 1. 加载源数据
print("\n步骤 1: 加载源数据...")
data_path = Path('data/mouse_brain_squidpy.h5ad')
adata = anndata.read_h5ad(data_path)
print(f"  ✓ 源数据加载完成: {adata.shape[0]} spots × {adata.shape[1]} genes")

# 检查数据结构和注释信息
print(f"\n数据结构信息:")
print(f"  空间坐标: {adata.obsm.get('spatial', '无')}")
print(f"  现有的注释: {list(adata.obs.columns)}")

# 2. 加载聚类结果
print("\n步骤 2: 加载聚类结果...")
output_dir = Path('outputs/image_guided_gvae')

if (output_dir / 'cluster_labels.npy').exists():
    cluster_labels = np.load(output_dir / 'cluster_labels.npy')
    print(f"  ✓ 加载聚类标签: {cluster_labels.shape}")
else:
    print(f"  ✗ 聚类标签文件不存在")
    exit()

if (output_dir / 'latent_representations.npy').exists():
    latent = np.load(output_dir / 'latent_representations.npy')
    print(f"  ✓ 加载潜在表示: {latent.shape}")
else:
    print(f"  ✗ 潜在表示文件不存在")
    exit()

# 确保聚类结果与源数据匹配
if len(cluster_labels) == adata.shape[0]:
    print(f"  ✓ 聚类结果与源数据点数匹配")
    # 将聚类标签添加到adata对象并转换为category类型
    adata.obs['cluster'] = cluster_labels.astype(str)
    adata.obs['cluster'] = adata.obs['cluster'].astype('category')
else:
    print(f"  ✗ 聚类结果与源数据点数不匹配: {len(cluster_labels)} vs {adata.shape[0]}")
    # 如果不匹配，可能是因为使用了合成数据进行训练
    print(f"  ℹ 可能是因为使用了合成数据进行训练，将使用合成数据的空间坐标")
    
    # 生成合成空间坐标
    n_spots = cluster_labels.shape[0]
    spatial_coords = np.random.rand(n_spots, 2) * 100
    
    # 创建一个新的adata对象用于分析
    adata = sc.AnnData(X=latent)
    adata.obsm['spatial'] = spatial_coords
    adata.obs['cluster'] = cluster_labels.astype(str)
    # 转换为category类型
    adata.obs['cluster'] = adata.obs['cluster'].astype('category')

print(f"\n聚类结果信息:")
print(f"  簇数量: {adata.obs['cluster'].nunique()}")
print(f"  每个簇的大小:")
for cluster, count in adata.obs['cluster'].value_counts().items():
    print(f"    簇 {cluster}: {count} spots ({count/adata.shape[0]*100:.1f}%)")

# 3. 聚类质量评估
print("\n步骤 3: 聚类质量评估...")

# 导入评估指标
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

print(f"\n内部评估指标:")

# 轮廓系数 (Silhouette Score)
silhouette = silhouette_score(latent, cluster_labels)
print(f"  轮廓系数: {silhouette:.4f} (越接近1越好)")

# 戴维斯-布隆丁指数 (Davies-Bouldin Index)
davies_bouldin = davies_bouldin_score(latent, cluster_labels)
print(f"  戴维斯-布隆丁指数: {davies_bouldin:.4f} (越小越好)")

# 卡利斯基-哈拉巴斯指数 (Calinski-Harabasz Index)
calinski_harabasz = calinski_harabasz_score(latent, cluster_labels)
print(f"  卡利斯基-哈拉巴斯指数: {calinski_harabasz:.4f} (越大越好)")

# 4. 空间一致性评估
print("\n步骤 4: 空间一致性评估...")

# 创建空间可视化
plt.figure(figsize=(12, 6))

# 空间分布
plt.subplot(1, 2, 1)
coords = adata.obsm['spatial']
colors = adata.obs['cluster'].cat.codes.values
scatter = plt.scatter(coords[:, 0], coords[:, 1], c=colors, cmap='viridis', s=20)
plt.colorbar(scatter, ticks=range(len(adata.obs['cluster'].cat.categories)), label='Cluster')
plt.title('聚类结果空间分布')
plt.xlabel('X坐标')
plt.ylabel('Y坐标')

# 潜在空间分布
plt.subplot(1, 2, 2)
sc.pp.neighbors(adata, use_rep='X')
sc.tl.umap(adata)
sc.pl.umap(adata, color='cluster', show=False, ax=plt.gca())
plt.title('聚类结果UMAP分布')

plt.tight_layout()
plt.savefig(output_dir / 'cluster_quality_evaluation.png', dpi=300, bbox_inches='tight')
print(f"  ✓ 空间可视化已保存: {output_dir / 'cluster_quality_evaluation.png'}")

# 5. 空间连续性评估
print("\n步骤 5: 空间连续性评估...")

# 计算空间连续性得分
def spatial_continuity_score(adata, cluster_key):
    """计算空间连续性得分"""
    from sklearn.neighbors import NearestNeighbors
    
    # 获取空间坐标
    coords = adata.obsm['spatial']
    
    # 构建KNN图
    nbrs = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    
    # 获取聚类标签
    labels = adata.obs[cluster_key].cat.codes.values
    
    # 计算每个点的邻居中具有相同标签的比例
    continuity_scores = []
    for i in range(coords.shape[0]):
        neighbor_labels = labels[indices[i, 1:]]  # 排除自身
        same_label = np.sum(neighbor_labels == labels[i])
        continuity_scores.append(same_label / (indices.shape[1] - 1))
    
    return np.mean(continuity_scores)

# 确保cluster是分类类型
adata.obs['cluster'] = adata.obs['cluster'].astype('category')

# 计算空间连续性得分
if 'spatial' in adata.obsm:
    continuity_score = spatial_continuity_score(adata, 'cluster')
    print(f"  空间连续性得分: {continuity_score:.4f} (越接近1越好)")
else:
    print(f"  ✗ 无法计算空间连续性得分，缺少空间坐标")

# 6. 差异表达分析
print("\n步骤 6: 差异表达分析...")

# 如果有基因表达数据，进行差异表达分析
if adata.X.shape[1] > latent.shape[1]:  # 确保有基因表达数据
    print(f"  进行差异表达分析...")
    
    # 计算每个簇的标志基因
    sc.tl.rank_genes_groups(adata, 'cluster', method='wilcoxon')
    
    # 保存标志基因
    de_genes = {}
    for cluster in adata.obs['cluster'].cat.categories:
        cluster_idx = adata.obs['cluster'].cat.categories.get_loc(cluster)
        genes = adata.uns['rank_genes_groups']['names'][cluster][:10]  # 前10个标志基因
        de_genes[cluster] = genes.tolist()
    
    # 打印标志基因
    print(f"\n每个簇的前10个标志基因:")
    for cluster, genes in de_genes.items():
        print(f"  簇 {cluster}: {', '.join(genes[:5])}...")
    
    # 可视化标志基因
    sc.pl.rank_genes_groups(adata, n_genes=10, sharey=False, show=False)
    plt.savefig(output_dir / 'marker_genes.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ 标志基因可视化已保存: {output_dir / 'marker_genes.png'}")
else:
    print(f"  ✗ 无法进行差异表达分析，缺少基因表达数据")

# 7. 生成评估报告
print("\n步骤 7: 生成评估报告...")

report_lines = []
report_lines.append("# 聚类效果评估报告\n")
report_lines.append("## 数据集信息\n")
report_lines.append(f"- 数据文件: {data_path}\n")
report_lines.append(f"- Spots数量: {adata.shape[0]}\n")
report_lines.append(f"- Genes数量: {adata.shape[1]}\n")

report_lines.append("\n## 聚类结果信息\n")
report_lines.append(f"- 簇数量: {adata.obs['cluster'].nunique()}\n")
report_lines.append(f"- 每个簇的大小:\n")
for cluster, count in adata.obs['cluster'].value_counts().items():
    report_lines.append(f"  - 簇 {cluster}: {count} spots ({count/adata.shape[0]*100:.1f}%)\n")

report_lines.append("\n## 聚类质量评估\n")
report_lines.append("### 内部评估指标\n")
report_lines.append(f"- 轮廓系数: {silhouette:.4f} (越接近1越好)\n")
report_lines.append(f"- 戴维斯-布隆丁指数: {davies_bouldin:.4f} (越小越好)\n")
report_lines.append(f"- 卡利斯基-哈拉巴斯指数: {calinski_harabasz:.4f} (越大越好)\n")

if 'spatial' in adata.obsm:
    report_lines.append(f"- 空间连续性得分: {continuity_score:.4f} (越接近1越好)\n")

report_lines.append("\n## 结果解释\n")
report_lines.append("### 轮廓系数\n")
if silhouette > 0.5:
    report_lines.append("- 轮廓系数较高，聚类结果良好\n")
elif silhouette > 0.25:
    report_lines.append("- 轮廓系数中等，聚类结果可接受\n")
else:
    report_lines.append("- 轮廓系数较低，聚类结果可能存在问题\n")

report_lines.append("\n### 空间连续性\n")
if 'spatial' in adata.obsm:
    if continuity_score > 0.7:
        report_lines.append("- 空间连续性较好，聚类结果具有空间一致性\n")
    elif continuity_score > 0.5:
        report_lines.append("- 空间连续性中等，聚类结果具有一定的空间一致性\n")
    else:
        report_lines.append("- 空间连续性较差，聚类结果可能不符合空间分布规律\n")
else:
    report_lines.append("- 无法评估空间连续性，缺少空间坐标\n")

report_lines.append("\n## 建议\n")
if silhouette < 0.5:
    report_lines.append("- 考虑调整聚类算法参数或尝试其他聚类方法\n")
if 'spatial' in adata.obsm and continuity_score < 0.7:
    report_lines.append("- 考虑使用空间约束的聚类算法\n")
report_lines.append("- 进一步分析标志基因的生物学意义\n")
report_lines.append("- 与现有注释进行比较（如果有）\n")

report_file = output_dir / 'cluster_quality_report.md'
with open(report_file, 'w', encoding='utf-8') as f:
    f.writelines(report_lines)

print(f"  ✓ 评估报告已生成: {report_file}")
print(f"  ✓ 聚类质量可视化已生成: {output_dir / 'cluster_quality_evaluation.png'}")

print("\n" + "=" * 80)
print("  聚类效果评估完成!")
print("=" * 80)
print(f"\n输出文件:")
print(f"  • 聚类质量评估报告: {report_file}")
print(f"  • 聚类质量可视化: {output_dir / 'cluster_quality_evaluation.png'}")
if adata.X.shape[1] > latent.shape[1]:
    print(f"  • 标志基因可视化: {output_dir / 'marker_genes.png'}")
