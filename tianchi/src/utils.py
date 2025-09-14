import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.metrics.pairwise import cosine_similarity
import warnings
from tqdm import tqdm
import os
import gc
import time

warnings.filterwarnings('ignore')


class ItemBasedCF:
    def __init__(self):
        self.user_item_matrix = None
        self.item_similarity = None
        self.item_index_map = None
        self.user_index_map = None
        self.item_ids = None
        self.user_ids = None
        self.target_items = None
        self.item_global_weights = {}

    def load_data_in_chunks(self, user_files, item_file, chunk_size=200000):
        """
        分段读取用户行为数据和目标商品集（无表头版本）
        增加chunk_size到200,000以提高处理速度
        """
        print("开始加载商品子集...")
        start_time = time.time()

        # 读取目标商品集（无表头）
        df_item = pd.read_csv(item_file, dtype=str, sep='\t', header=None)
        # 假设第一列是item_id
        self.target_items = set(df_item[0].unique())
        print(f"商品子集加载完成，共 {len(self.target_items)} 个商品，耗时 {time.time() - start_time:.2f} 秒")

        # 初始化数据结构
        all_user_ids = set()
        all_item_ids = set(self.target_items)  # 包含所有目标商品

        # 使用字典存储用户-商品的最大权重，减少内存使用
        user_item_weights = {}

        # 处理每个用户数据文件
        for user_file in user_files:
            print(f"处理文件: {user_file}")
            file_start_time = time.time()

            # 获取文件行数用于进度条
            try:
                with open(user_file, 'r', encoding='utf-8') as f:
                    total_rows = sum(1 for _ in f)
            except:
                total_rows = 1000000  # 如果无法获取行数，使用默认值

            # 分块读取数据（无表头）
            chunk_iter = pd.read_csv(
                user_file,
                dtype=str,
                sep='\t',
                chunksize=chunk_size,
                header=None
            )

            # 计算行为权重：购买(4)>加购物车(3)>收藏(2)>浏览(1)
            behavior_weights = {'1': 1, '2': 2, '3': 3, '4': 4}

            # 使用tqdm显示进度条
            for chunk_idx, chunk in enumerate(tqdm(chunk_iter, total=total_rows / chunk_size, desc="处理数据块")):
                # 假设列顺序：user_id, item_id, behavior_type, user_geohash, item_category, time
                # 过滤出目标商品
                chunk = chunk[chunk[1].isin(self.target_items)]

                if chunk.empty:
                    continue

                # 计算行为权重
                chunk['behavior_weight'] = chunk[2].map(behavior_weights)

                # 按用户和商品分组，取最大权重
                chunk_agg = chunk.groupby([0, 1])['behavior_weight'].max().reset_index()

                # 收集数据
                for _, row in chunk_agg.iterrows():
                    user_id, item_id, weight = row[0], row[1], row['behavior_weight']
                    all_user_ids.add(user_id)

                    # 更新商品全局权重
                    self.item_global_weights[item_id] = self.item_global_weights.get(item_id, 0) + weight

                    # 使用字典存储用户-商品权重，减少内存使用
                    if user_id not in user_item_weights:
                        user_item_weights[user_id] = {}
                    user_item_weights[user_id][item_id] = max(
                        user_item_weights[user_id].get(item_id, 0),
                        weight
                    )

                # 定期清理内存
                if chunk_idx % 20 == 0:
                    gc.collect()

            print(f"文件 {user_file} 处理完成，耗时 {time.time() - file_start_time:.2f} 秒")

        # 创建用户和物品的索引映射
        self.user_ids = sorted(all_user_ids)
        self.item_ids = sorted(all_item_ids)
        self.user_index_map = {user: idx for idx, user in enumerate(self.user_ids)}
        self.item_index_map = {item: idx for idx, item in enumerate(self.item_ids)}

        print(f"用户行为数据处理完成，共 {len(self.user_ids)} 个用户，{len(self.item_ids)} 个物品")
        print(f"总耗时: {time.time() - start_time:.2f} 秒")

        # 构建稀疏矩阵
        rows, cols, data = [], [], []
        for user_id, items in tqdm(user_item_weights.items(), desc="构建稀疏矩阵"):
            user_idx = self.user_index_map[user_id]
            for item_id, weight in items.items():
                item_idx = self.item_index_map[item_id]
                rows.append(user_idx)
                cols.append(item_idx)
                data.append(weight)

        self.user_item_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(len(self.user_ids), len(self.item_ids))
        )

        # 清理内存
        del user_item_weights
        gc.collect()

        return True

    def calculate_similarity(self, min_interactions=2, similarity_threshold=0.01):
        """
        计算物品之间的余弦相似度
        降低min_interactions和similarity_threshold以增加推荐覆盖率
        """
        print("开始计算物品相似度...")
        start_time = time.time()

        # 计算每个物品的交互次数
        item_interactions = np.array(self.user_item_matrix.astype(bool).sum(axis=0)).flatten()

        # 过滤交互次数过少的物品
        valid_items = np.where(item_interactions >= min_interactions)[0]
        print(f"有效物品数量: {len(valid_items)} (交互次数 >= {min_interactions})")

        if len(valid_items) == 0:
            print("没有足够的有效物品计算相似度")
            return False

        # 计算余弦相似度 - 分块处理以避免内存不足
        n_items = len(valid_items)
        block_size = 1000  # 增加块大小到1000以提高处理速度
        self.item_similarity = lil_matrix((len(self.item_ids), len(self.item_ids)))

        for i in tqdm(range(0, n_items, block_size), desc="计算相似度"):
            end_idx = min(i + block_size, n_items)
            block_indices = valid_items[i:end_idx]
            block_matrix = self.user_item_matrix[:, block_indices]

            # 计算当前块与所有物品的相似度
            block_similarity = cosine_similarity(block_matrix.T, self.user_item_matrix.T, dense_output=False)

            # 只保留相似度高于阈值的值
            block_similarity.data[block_similarity.data < similarity_threshold] = 0
            block_similarity.eliminate_zeros()

            # 将相似度映射回原始物品索引
            for block_idx, orig_idx in enumerate(block_indices):
                row_data = block_similarity.getrow(block_idx).toarray().flatten()
                nonzero_indices = np.where(row_data > 0)[0]

                for j in nonzero_indices:
                    if orig_idx != j and row_data[j] > 0:
                        self.item_similarity[orig_idx, j] = row_data[j]

        # 转换为CSR格式以提高后续操作效率
        self.item_similarity = self.item_similarity.tocsr()
        print(f"相似度计算完成，耗时 {time.time() - start_time:.2f} 秒")
        return True

    def generate_recommendation(self, user_id):
        """
        为指定用户生成单个推荐
        """
        if user_id not in self.user_index_map:
            # 如果用户不在训练集中，返回全局最热门的商品
            if self.item_global_weights:
                return max(self.item_global_weights.items(), key=lambda x: x[1])[0]
            else:
                return self.item_ids[0] if self.item_ids else ""

        user_idx = self.user_index_map[user_id]
        user_interactions = self.user_item_matrix[user_idx, :].toarray().flatten()

        # 获取用户已交互的物品索引
        interacted_indices = np.where(user_interactions > 0)[0]

        if len(interacted_indices) == 0:
            # 如果用户没有交互历史，返回全局最热门的商品
            if self.item_global_weights:
                return max(self.item_global_weights.items(), key=lambda x: x[1])[0]
            else:
                return self.item_ids[0] if self.item_ids else ""

        # 初始化预测评分
        predictions = np.zeros(len(self.item_ids))

        for item_idx in interacted_indices:
            # 获取当前物品的相似度向量
            similarities = self.item_similarity[item_idx, :].toarray().flatten()

            # 获取当前物品的交互权重
            interaction_weight = user_interactions[item_idx]

            # 累加预测评分
            predictions += similarities * interaction_weight

        # 排除用户已经交互过的物品
        predictions[interacted_indices] = 0

        # 获取最高预测分数的物品
        if np.max(predictions) > 0:
            best_item_idx = np.argmax(predictions)
            return self.item_ids[best_item_idx]
        else:
            # 如果没有找到推荐，返回用户最近交互的商品
            last_interacted_idx = interacted_indices[-1]
            return self.item_ids[last_interacted_idx]

    def generate_submission(self, user_files, item_file, output_file):
        """
        生成提交文件
        """
        # 加载数据
        if not self.load_data_in_chunks(user_files, item_file):
            print("数据加载失败")
            return None

        # 计算相似度
        if not self.calculate_similarity(min_interactions=2, similarity_threshold=0.01):
            print("相似度计算失败")
            return None

        # 生成推荐结果
        results = []
        print("生成推荐结果...")
        start_time = time.time()

        for user in tqdm(self.user_ids, desc="为用户生成推荐"):
            recommended_item = self.generate_recommendation(user)
            results.append((user, recommended_item))

        # 保存结果 - 按照要求的格式
        with open(output_file, 'w', encoding='utf-8') as f:
            for user_id, item_id in results:
                f.write(f"{user_id}\t{item_id}\n")

        print(f"生成推荐结果完成，共{len(results)}条推荐记录，耗时 {time.time() - start_time:.2f} 秒")
        return results


# 主程序
if __name__ == "__main__":
    # 初始化推荐系统
    recommender = ItemBasedCF()

    # 文件路径设置 - 根据图片中的文件名
    data_dir = "../data"  # 数据目录
    user_files = [
        os.path.join(data_dir, "tianchi_fresh_comp_train_user_online_partA.txt"),
        os.path.join(data_dir, "tianchi_fresh_comp_train_user_online_partB.txt")
    ]
    item_file = os.path.join(data_dir, "tianchi_fresh_comp_train_item_online.txt")
    output_file = os.path.join(data_dir, "tianchi_mobile_recommendation_predict.txt")

    # 检查文件是否存在
    for file in user_files + [item_file]:
        if not os.path.exists(file):
            print(f"警告: 文件 {file} 不存在")

    # 生成推荐结果
    results = recommender.generate_submission(user_files, item_file, output_file)

    if results is not None:
        # 打印前10条推荐结果
        print("\n推荐结果示例:")
        for i in range(min(10, len(results))):
            print(f"{results[i][0]}\t{results[i][1]}")
    else:
        print("推荐结果生成失败")
# import pandas as pd
# import numpy as np
# from collections import defaultdict
# import time
# import gc
# from scipy.sparse import lil_matrix, csr_matrix
# import config
#
#
# def log_message(message):
#     """打印带时间戳的日志信息"""
#     print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")
#
#
# def load_item_subset():
#     """加载商品子集P"""
#     log_message("开始加载商品子集...")
#     try:
#         items_df = pd.read_csv(config.ITEM_DATA_PATH, sep='\t', header=None,
#                                names=['item_id', 'item_geohash', 'item_category'])
#         item_subset = set(items_df['item_id'].astype(str).values)
#         log_message(f"商品子集加载完成，共 {len(item_subset)} 个商品")
#         return item_subset
#     except Exception as e:
#         log_message(f"加载商品子集时出错: {str(e)}")
#         return set()
#
#
# def process_user_data_chunk(chunk, user_items, item_users, item_popularity):
#     """处理用户数据块，更新用户-物品和物品-用户关系"""
#     # 确保列名正确
#     if len(chunk.columns) < 5:
#         chunk.columns = ['user_id', 'item_id', 'behavior_type', 'user_geohash', 'time']
#
#     # 转换数据类型
#     chunk['user_id'] = chunk['user_id'].astype(str)
#     chunk['item_id'] = chunk['item_id'].astype(str)
#     chunk['behavior_type'] = chunk['behavior_type'].astype(str)
#
#     # 应用行为权重
#     chunk['weight'] = chunk['behavior_type'].map(
#         lambda x: config.BEHAVIOR_WEIGHTS.get(x, 1)
#     )
#
#     # 更新用户-物品关系
#     for _, row in chunk.iterrows():
#         user_id = row['user_id']
#         item_id = row['item_id']
#         weight = row['weight']
#
#         # 更新用户物品关系
#         if user_id not in user_items:
#             user_items[user_id] = defaultdict(float)
#         user_items[user_id][item_id] = max(user_items[user_id][item_id], weight)
#
#         # 更新物品用户关系
#         if item_id not in item_users:
#             item_users[item_id] = defaultdict(float)
#         item_users[item_id][user_id] = max(item_users[item_id][user_id], weight)
#
#         # 更新物品流行度
#         item_popularity[item_id] = item_popularity.get(item_id, 0) + 1
#
#     return user_items, item_users, item_popularity
#
#
# def calculate_item_similarity(item_users, item_popularity):
#     """计算物品相似度矩阵"""
#     log_message("开始计算物品相似度...")
#
#     # 获取所有物品ID
#     all_items = list(item_users.keys())
#     n_items = len(all_items)
#     item_to_idx = {item: idx for idx, item in enumerate(all_items)}
#
#     # 初始化相似度矩阵
#     if config.USE_SPARSE_MATRIX:
#         similarity_matrix = lil_matrix((n_items, n_items), dtype=np.float32)
#     else:
#         similarity_matrix = np.zeros((n_items, n_items), dtype=np.float32)
#
#     # 计算共现矩阵
#     cooccurrence_matrix = lil_matrix((n_items, n_items), dtype=np.int32)
#
#     # 遍历所有用户，计算物品共现
#     for user, items in item_users.items():
#         item_list = list(items.keys())
#         for i in range(len(item_list)):
#             item_i = item_list[i]
#             idx_i = item_to_idx[item_i]
#             for j in range(i + 1, len(item_list)):
#                 item_j = item_list[j]
#                 idx_j = item_to_idx[item_j]
#                 cooccurrence_matrix[idx_i, idx_j] += 1
#                 cooccurrence_matrix[idx_j, idx_i] += 1
#
#     # 计算余弦相似度
#     for i in range(n_items):
#         item_i = all_items[i]
#         popularity_i = item_popularity.get(item_i, 1)
#
#         for j in range(i + 1, n_items):
#             item_j = all_items[j]
#             cooccurrence = cooccurrence_matrix[i, j]
#
#             if cooccurrence > 0:
#                 popularity_j = item_popularity.get(item_j, 1)
#                 similarity = cooccurrence / np.sqrt(popularity_i * popularity_j)
#
#                 if similarity >= config.SIMILARITY_THRESHOLD:
#                     similarity_matrix[i, j] = similarity
#                     similarity_matrix[j, i] = similarity
#
#     log_message("物品相似度计算完成")
#     return similarity_matrix, item_to_idx, all_items
#
#
# def generate_recommendations(user_items, similarity_matrix, item_to_idx, all_items, item_subset):
#     """为每个用户生成推荐列表"""
#     log_message("开始生成推荐列表...")
#     recommendations = {}
#
#     for user_id, items in user_items.items():
#         user_recommendations = defaultdict(float)
#
#         # 获取用户交互过的物品
#         interacted_items = list(items.keys())
#
#         # 为每个交互过的物品寻找相似物品
#         for item_id in interacted_items:
#             if item_id not in item_to_idx:
#                 continue
#
#             item_idx = item_to_idx[item_id]
#             weight = items[item_id]
#
#             # 获取相似物品
#             similar_items = similarity_matrix[item_idx].toarray().flatten() if config.USE_SPARSE_MATRIX else \
#             similarity_matrix[item_idx]
#
#             for j, similarity in enumerate(similar_items):
#                 if similarity > 0:
#                     similar_item_id = all_items[j]
#                     # 只推荐在商品子集P中且用户未交互过的商品
#                     if similar_item_id in item_subset and similar_item_id not in interacted_items:
#                         user_recommendations[similar_item_id] += similarity * weight
#
#         # 排序并取Top-N
#         sorted_recommendations = sorted(
#             user_recommendations.items(), key=lambda x: x[1], reverse=True
#         )[:config.TOP_N_RECOMMENDATIONS]
#
#         recommendations[user_id] = sorted_recommendations
#
#     log_message("推荐列表生成完成")
#     return recommendations
#
#
# def save_recommendations(recommendations, output_path):
#     """保存推荐结果到文件"""
#     log_message("开始保存推荐结果...")
#     with open(output_path, 'w', encoding='utf-8') as f:
#         f.write("user_id\titem_id\n")
#         for user_id, items in recommendations.items():
#             for item_id, score in items:
#                 f.write(f"{user_id}\t{item_id}\n")
#     log_message(f"推荐结果已保存到 {output_path}")