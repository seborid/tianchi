import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import os

# 配置参数
CHUNK_SIZE = 100000
BEHAVIOR_WEIGHTS = {1: 1.0, 2: 2.0, 3: 3.0, 4: 5.0}


def build_sparse_user_item_matrix(user_item_interactions):
    """
    构建稀疏用户-物品矩阵
    """
    print("构建稀疏用户-物品矩阵...")

    # 获取所有用户和物品
    all_users = list(user_item_interactions.keys())
    all_items = set()

    for user_items in user_item_interactions.values():
        all_items.update(user_items.keys())

    all_items = list(all_items)

    print(f"共有 {len(all_users)} 个用户和 {len(all_items)} 个物品")

    # 创建映射
    user_idx_map = {user: idx for idx, user in enumerate(all_users)}
    item_idx_map = {item: idx for idx, item in enumerate(all_items)}

    # 使用稀疏矩阵格式
    row_indices = []
    col_indices = []
    data = []

    for user, items in user_item_interactions.items():
        user_idx = user_idx_map[user]
        for item, score in items.items():
            item_idx = item_idx_map[item]
            row_indices.append(user_idx)
            col_indices.append(item_idx)
            data.append(score)

    # 创建稀疏矩阵
    sparse_matrix = csr_matrix((data, (row_indices, col_indices)),
                               shape=(len(all_users), len(all_items)))

    return sparse_matrix, all_users, all_items


def calculate_user_similarity_sparse(sparse_matrix, sample_size=10000):
    """
    使用采样方法计算用户相似度
    """
    print("使用采样方法计算用户相似度...")

    # 随机采样一部分用户
    n_users = sparse_matrix.shape[0]
    if n_users > sample_size:
        sampled_indices = np.random.choice(n_users, sample_size, replace=False)
        sampled_matrix = sparse_matrix[sampled_indices, :]
    else:
        sampled_matrix = sparse_matrix

    # 计算采样用户的相似度
    similarity_matrix = cosine_similarity(sampled_matrix)

    return similarity_matrix, sampled_indices if n_users > sample_size else np.arange(n_users)


def main():
    # 文件路径
    data_dir = "../data/"
    user_file_paths = [
        os.path.join(data_dir, "tianchi_fresh_comp_train_user_online_partA.txt"),
        os.path.join(data_dir, "tianchi_fresh_comp_train_user_online_partB.txt")
    ]
    item_file_path = os.path.join(data_dir, "tianchi_fresh_comp_train_item_online.txt")

    # 检查文件是否存在
    for path in user_file_paths + [item_file_path]:
        if not os.path.exists(path):
            print(f"错误: 文件 {path} 不存在")
            return

    # 读取商品子集
    item_df = pd.read_csv(
        item_file_path,
        sep='\t',
        header=None,
        usecols=[0, 1],
        names=['item_id', 'item_category']
    )
    target_items = set(item_df['item_id'].unique())
    print(f"商品子集中有 {len(target_items)} 个商品")

    # 初始化用户-物品交互字典
    user_item_interactions = defaultdict(lambda: defaultdict(float))

    # 处理每个用户行为文件
    for user_file_path in user_file_paths:
        print(f"处理用户行为文件: {user_file_path}")

        # 使用分块读取
        chunk_iterator = pd.read_csv(
            user_file_path,
            sep='\t',
            header=None,
            usecols=[0, 1, 2],
            names=['user_id', 'item_id', 'behavior_type'],
            chunksize=CHUNK_SIZE
        )

        for i, chunk in enumerate(chunk_iterator):
            print(f"处理第 {i + 1} 个数据块...")

            # 过滤只保留目标商品集中的物品
            chunk = chunk[chunk['item_id'].isin(target_items)]

            # 处理每个数据块
            for _, row in chunk.iterrows():
                user_id = row['user_id']
                item_id = row['item_id']
                behavior_type = row['behavior_type']

                # 添加加权行为得分
                user_item_interactions[user_id][item_id] += BEHAVIOR_WEIGHTS.get(behavior_type, 1.0)

    # 构建稀疏用户-物品矩阵
    sparse_matrix, all_users, all_items = build_sparse_user_item_matrix(user_item_interactions)

    # 计算用户相似度（使用采样）
    similarity_matrix, sampled_indices = calculate_user_similarity_sparse(sparse_matrix, sample_size=10000)

    # 生成推荐（这里需要根据您的具体需求实现）
    # 由于数据量极大，推荐生成策略也需要相应调整

    print("处理完成")


if __name__ == "__main__":
    main()
# import pandas as pd
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from collections import defaultdict
# import os
#
# # 配置参数
# CHUNK_SIZE = 10000000  # 每次读取的数据块大小
# BEHAVIOR_WEIGHTS = {1: 1.0, 2: 2.0, 3: 3.0, 4: 5.0}  # 行为权重
#
#
# def load_data_in_chunks(user_file_paths, item_file_path):
#     """
#     分段读取用户行为数据
#     """
#     print("正在读取商品文件...")
#
#     # 读取商品子集 - 根据日志，商品文件有2列，但被识别为3列（多了一个空列）
#     # 我们明确指定列名和列数
#     item_df = pd.read_csv(
#         item_file_path,
#         sep='\t',
#         header=None,
#         usecols=[0, 1],  # 只取前两列
#         names=['item_id', 'item_category']
#     )
#     target_items = set(item_df['item_id'].unique())
#     print(f"商品子集中有 {len(target_items)} 个商品")
#
#     # 初始化用户-物品交互字典
#     user_item_interactions = defaultdict(lambda: defaultdict(float))
#
#     # 处理每个用户行为文件
#     for user_file_path in user_file_paths:
#         print(f"处理用户行为文件: {user_file_path}")
#
#         # 使用分块读取 - 根据日志，用户行为文件有5列，但被识别为6列（多了一个空列）
#         # 我们明确指定列名和列数
#         chunk_iterator = pd.read_csv(
#             user_file_path,
#             sep='\t',
#             header=None,
#             usecols=[0, 1, 2, 3, 4],  # 只取前5列
#             names=['user_id', 'item_id', 'behavior_type', 'item_category', 'time'],
#             chunksize=CHUNK_SIZE
#         )
#
#         for i, chunk in enumerate(chunk_iterator):
#             print(f"处理第 {i + 1} 个数据块...")
#
#             # 过滤只保留目标商品集中的物品
#             chunk = chunk[chunk['item_id'].isin(target_items)]
#
#             if len(chunk) == 0:
#                 print(f"警告: 第 {i + 1} 个数据块过滤后没有数据")
#                 continue
#
#             # 处理每个数据块
#             for _, row in chunk.iterrows():
#                 user_id = row['user_id']
#                 item_id = row['item_id']
#                 behavior_type = row['behavior_type']
#
#                 # 添加加权行为得分
#                 user_item_interactions[user_id][item_id] += BEHAVIOR_WEIGHTS.get(behavior_type, 1.0)
#
#     return user_item_interactions, target_items
#
#
# def build_user_item_matrix(user_item_interactions):
#     """
#     从交互字典构建用户-物品矩阵
#     """
#     print("构建用户-物品矩阵...")
#
#     # 检查是否有交互数据
#     if not user_item_interactions:
#         print("错误: 没有用户-物品交互数据")
#         return np.array([]), [], []
#
#     # 获取所有用户和物品
#     all_users = list(user_item_interactions.keys())
#     all_items = set()
#
#     for user_items in user_item_interactions.values():
#         all_items.update(user_items.keys())
#
#     all_items = list(all_items)
#
#     print(f"共有 {len(all_users)} 个用户和 {len(all_items)} 个物品")
#
#     if len(all_users) == 0 or len(all_items) == 0:
#         print("错误: 用户或物品数量为0")
#         return np.array([]), [], []
#
#     # 创建用户-物品映射
#     user_idx_map = {user: idx for idx, user in enumerate(all_users)}
#     item_idx_map = {item: idx for idx, item in enumerate(all_items)}
#
#     # 构建矩阵
#     matrix = np.zeros((len(all_users), len(all_items)))
#
#     for user, items in user_item_interactions.items():
#         user_idx = user_idx_map[user]
#         for item, score in items.items():
#             item_idx = item_idx_map[item]
#             matrix[user_idx][item_idx] = score
#
#     return matrix, all_users, all_items
#
#
# def calculate_user_similarity(user_item_matrix):
#     """
#     计算用户相似度矩阵
#     """
#     print("计算用户相似度...")
#
#     # 检查矩阵是否为空
#     if user_item_matrix.size == 0:
#         print("错误: 用户-物品矩阵为空")
#         return np.array([])
#
#     similarity_matrix = cosine_similarity(user_item_matrix)
#     return similarity_matrix
#
#
# def generate_recommendations_chunked(user_item_matrix, similarity_matrix, all_users, all_items, target_items, top_n=10):
#     """
#     分段生成推荐结果
#     """
#     print("生成推荐结果...")
#
#     # 检查输入是否有效
#     if user_item_matrix.size == 0 or similarity_matrix.size == 0:
#         print("错误: 输入矩阵为空")
#         return {}
#
#     recommendations = {}
#
#     # 获取目标物品的索引
#     target_item_idxs = [i for i, item in enumerate(all_items) if item in target_items]
#
#     if not target_item_idxs:
#         print("错误: 没有目标物品的索引")
#         return {}
#
#     # 分批处理用户
#     batch_size = 1000
#     num_users = len(all_users)
#
#     for start_idx in range(0, num_users, batch_size):
#         end_idx = min(start_idx + batch_size, num_users)
#         print(f"处理用户 {start_idx} 到 {end_idx - 1}...")
#
#         for user_idx in range(start_idx, end_idx):
#             user_id = all_users[user_idx]
#
#             # 获取当前用户的交互向量
#             user_interactions = user_item_matrix[user_idx]
#
#             # 找到最相似的K个用户
#             user_similarities = similarity_matrix[user_idx]
#             similar_user_idxs = np.argsort(user_similarities)[-top_n - 1:-1][::-1]  # 排除自己
#
#             # 计算预测得分
#             predicted_scores = defaultdict(float)
#
#             for sim_user_idx in similar_user_idxs:
#                 similarity_score = user_similarities[sim_user_idx]
#                 sim_user_interactions = user_item_matrix[sim_user_idx]
#
#                 # 只考虑目标物品且当前用户没有交互过的物品
#                 for item_idx in target_item_idxs:
#                     if user_interactions[item_idx] == 0 and sim_user_interactions[item_idx] > 0:
#                         item_id = all_items[item_idx]
#                         predicted_scores[item_id] += similarity_score * sim_user_interactions[item_idx]
#
#             # 按预测得分排序，获取Top-N推荐
#             user_recommendations = sorted(
#                 [(item, score) for item, score in predicted_scores.items()],
#                 key=lambda x: x[1],
#                 reverse=True
#             )[:top_n]
#
#             recommendations[user_id] = [item for item, score in user_recommendations]
#
#     return recommendations
#
#
# def main():
#     # 文件路径（根据图片中的文件名）
#     data_dir = "../data/"
#     user_file_paths = [
#         os.path.join(data_dir, "tianchi_fresh_comp_train_user_online_partA.txt"),
#         os.path.join(data_dir, "tianchi_fresh_comp_train_user_online_partB.txt")
#     ]
#     item_file_path = os.path.join(data_dir, "tianchi_fresh_comp_train_item_online.txt")
#
#     # 检查文件是否存在
#     for path in user_file_paths + [item_file_path]:
#         if not os.path.exists(path):
#             print(f"错误: 文件 {path} 不存在")
#             return
#
#     # 分段读取数据
#     user_item_interactions, target_items = load_data_in_chunks(user_file_paths, item_file_path)
#
#     # 检查是否有数据
#     if not user_item_interactions:
#         print("错误: 没有用户-物品交互数据，无法继续")
#         return
#
#     # 构建用户-物品矩阵
#     user_item_matrix, all_users, all_items = build_user_item_matrix(user_item_interactions)
#
#     # 检查矩阵是否有效
#     if user_item_matrix.size == 0:
#         print("错误: 用户-物品矩阵为空，无法继续")
#         return
#
#     # 计算用户相似度
#     similarity_matrix = calculate_user_similarity(user_item_matrix)
#
#     # 检查相似度矩阵是否有效
#     if similarity_matrix.size == 0:
#         print("错误: 相似度矩阵为空，无法继续")
#         return
#
#     # 生成推荐
#     recommendations = generate_recommendations_chunked(
#         user_item_matrix, similarity_matrix, all_users, all_items, target_items, top_n=20
#     )
#
#     # 准备输出结果
#     output_lines = []
#     for user, items in recommendations.items():
#         for item in items:
#             output_lines.append(f"{user}\t{item}")
#
#     # 保存结果到文件
#     with open("recommendation_result.txt", "w") as f:
#         f.write("\n".join(output_lines))
#
#     print(f"生成推荐完成，共{len(output_lines)}条推荐记录")
#
#
# # 运行主函数
# if __name__ == "__main__":
#     main()