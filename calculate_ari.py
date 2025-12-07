import numpy as np
from sklearn.metrics import adjusted_rand_score
from pathlib import Path

print("=" * 80)
print("  计算聚类的调整兰德指数 (ARI)")
print("=" * 80)

# 1. 加载聚类结果和真实标签
print("\n步骤 1: 加载数据...")
output_dir = Path('outputs/image_guided_gvae')

# 加载聚类标签
cluster_labels_path = output_dir / 'cluster_labels.npy'
if cluster_labels_path.exists():
    cluster_labels = np.load(cluster_labels_path)
    print(f"  ✓ 加载聚类标签: {cluster_labels.shape}")
else:
    print(f"  ✗ 聚类标签文件不存在: {cluster_labels_path}")
    exit()

# 加载真实标签
true_labels_path = output_dir / 'true_labels.npy'
if true_labels_path.exists():
    # 使用allow_pickle=True加载可能包含字符串的标签
    true_labels = np.load(true_labels_path, allow_pickle=True)
    print(f"  ✓ 加载真实标签: {true_labels.shape}")
else:
    print(f"  ✗ 真实标签文件不存在: {true_labels_path}")
    exit()

# 2. 检查数据形状匹配
print("\n步骤 2: 检查数据匹配...")
if cluster_labels.shape != true_labels.shape:
    print(f"  ✗ 标签形状不匹配: 聚类标签 {cluster_labels.shape}, 真实标签 {true_labels.shape}")
    exit()
else:
    print(f"  ✓ 标签形状匹配: {cluster_labels.shape}")

# 3. 处理可能的字符串标签
print("\n步骤 3: 处理标签类型...")
print(f"  聚类标签类型: {cluster_labels.dtype}")
print(f"  真实标签类型: {true_labels.dtype}")

# 如果真实标签是字符串类型，转换为整数编码
if true_labels.dtype == 'object':
    print(f"  真实标签是字符串类型，正在转换为整数编码...")
    unique_labels, encoded_labels = np.unique(true_labels, return_inverse=True)
    true_labels = encoded_labels
    print(f"  转换后真实标签类型: {true_labels.dtype}")
    print(f"  唯一标签数量: {len(unique_labels)}")

# 4. 计算ARI
print("\n步骤 4: 计算ARI...")
ari = adjusted_rand_score(true_labels, cluster_labels)
print(f"\n  调整兰德指数 (ARI): {ari:.4f}")

# 5. 解释结果
print("\n步骤 5: 结果解释...")
if ari == 1.0:
    print("  完美匹配: 聚类结果与真实标签完全一致")
elif ari >= 0.8:
    print("  优秀匹配: 聚类结果与真实标签高度一致")
elif ari >= 0.6:
    print("  良好匹配: 聚类结果与真实标签有较好的一致性")
elif ari >= 0.4:
    print("  中等匹配: 聚类结果与真实标签有一定的一致性")
elif ari >= 0.2:
    print("  弱匹配: 聚类结果与真实标签有较弱的一致性")
else:
    print("  随机匹配: 聚类结果与真实标签基本一致，可能是随机分配的结果")

print(f"\n" + "=" * 80)
print("  ARI计算完成!")
print("=" * 80)
