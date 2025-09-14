import math
from collections import defaultdict
import heapq
import time

# 定义文件路径
item_file = '../data/tianchi_fresh_comp_train_item_online.txt'
user_file_a = '../data/tianchi_fresh_comp_train_user_online_partA.txt'
user_file_b = '../data/tianchi_fresh_comp_train_user_online_partB.txt'
output_file = '../data/recommendation_result_low_1.txt'


def log_message(message):
    """打印带时间戳的日志信息"""
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}")


# 读取商品子集P
log_message("开始加载商品子集P...")
P_items = set()
with open(item_file, 'r') as f:
    for line in f:
        item_id = line.strip().split('\t')[0]
        P_items.add(item_id)
log_message(f"已加载商品子集P，商品数量: {len(P_items)}")

# 提取购买行为记录
log_message("开始处理用户行为文件...")
user_purchases = defaultdict(set)  # user_id -> set of purchased item_ids
item_users = defaultdict(set)  # item_id -> set of user_ids who purchased it


def process_user_file(file_path):
    line_count = 0
    purchase_count = 0
    with open(file_path, 'r') as f:
        for line in f:
            line_count += 1
            if line_count % 10000000 == 0:
                log_message(f"处理文件中: {file_path}, 已处理行数: {line_count}, 购买记录: {purchase_count}")
            data = line.strip().split('\t')
            if len(data) < 6:
                continue
            user_id = data[0]
            item_id = data[1]
            behavior_type = data[2]
            time_str = data[5]
            if behavior_type == '4':  # 只处理购买行为
                date_str = time_str.split(' ')[0]
                if date_str >= '2014-11-18' and date_str <= '2014-12-18':
                    user_purchases[user_id].add(item_id)
                    item_users[item_id].add(user_id)
                    purchase_count += 1
    log_message(f"完成处理文件: {file_path}, 总行数: {line_count}, 购买记录: {purchase_count}")


# 处理两个用户行为文件
process_user_file(user_file_a)
process_user_file(user_file_b)

log_message(f"用户购买记录数量: {len(user_purchases)}")
log_message(f"商品被购买记录数量: {len(item_users)}")

# 构建共现矩阵：只考虑P中的物品
log_message("开始构建共现矩阵...")
cooccurrence = defaultdict(lambda: defaultdict(int))
user_count = 0
for user_id, items_set in user_purchases.items():
    user_count += 1
    if user_count % 100000 == 0:
        log_message(f"已处理用户数: {user_count}")

    items_list = list(items_set)
    for i in items_list:
        if i in P_items:  # 只考虑P中的物品
            for j in items_list:
                if j != i:
                    cooccurrence[i][j] += 1

# 计算相似度矩阵：使用改进的余弦相似度
log_message("开始计算相似度矩阵...")
sim_dict = defaultdict(dict)  # sim_dict[i][j] = similarity between i and j for i in P
# min_similarity = 0.1  # 设置相似度阈值，过滤低相似度物品

for i in P_items:
    if i in cooccurrence:
        for j, co_count in cooccurrence[i].items():
            if i in item_users and j in item_users:
                len_i = len(item_users[i])
                len_j = len(item_users[j])
                if len_i > 0 and len_j > 0:
                    sim_val = co_count / math.sqrt(len_i * len_j)
                    # if sim_val >= min_similarity:  # 只保留相似度较高的物品对
                    #     sim_dict[i][j] = sim_val
                    sim_dict[i][j] = sim_val

# 构建反向映射：对于每个物品j，存储与P中物品i的相似度
log_message("开始构建反向映射...")
j_sim = defaultdict(list)
for i in P_items:
    if i in sim_dict:
        for j, sim_val in sim_dict[i].items():
            j_sim[j].append((i, sim_val))

# 计算每个用户的推荐分数
log_message("开始计算用户推荐分数...")
user_scores = defaultdict(lambda: defaultdict(float))
user_count = 0
for user_id, purchased_items in user_purchases.items():
    user_count += 1
    if user_count % 100000 == 0:
        log_message(f"已计算用户推荐分数: {user_count}")

    for j in purchased_items:
        if j in j_sim:
            for (i, sim_val) in j_sim[j]:
                user_scores[user_id][i] += sim_val

# 为每个用户选择Top-K物品作为推荐，并设置分数阈值
log_message("开始生成推荐结果...")
recommendations = []
min_score = 0  # 设置推荐分数阈值
max_recommendations_per_user = 6  # 每个用户最多推荐3个物品

for user_id, scores in user_scores.items():
    # 使用堆来获取分数最高的物品，但最多不超过max_recommendations_per_user
    top_items = heapq.nlargest(max_recommendations_per_user, scores.items(), key=lambda x: x[1])
    for item_id, score in top_items:
        if score >= min_score:  # 只保留分数较高的推荐
            recommendations.append((user_id, item_id))

# 输出结果到文件
log_message("开始写入推荐结果...")
with open(output_file, 'w') as f:
    for user_id, item_id in recommendations:
        f.write(f"{user_id}\t{item_id}\n")

log_message(f"推荐结果已写入文件: {output_file}")
log_message(f"共生成{len(recommendations)}条推荐记录")
log_message("程序执行完毕")
# import math
# from collections import defaultdict
# import heapq
#
# # 定义文件路径
# item_file = '../data/tianchi_fresh_comp_train_item_online.txt'
# user_file_a = '../data/tianchi_fresh_comp_train_user_online_partA.txt'
# user_file_b = '../data/tianchi_fresh_comp_train_user_online_partB.txt'
# output_file = '../data/recommendation_result.txt'
#
# # 读取商品子集P
# P_items = set()
# with open(item_file, 'r') as f:
#     for line in f:
#         item_id = line.strip().split('\t')[0]
#         P_items.add(item_id)
#
# # 提取购买行为记录：只保留behavior_type=4且时间在2014-11-18至2014-12-18之间的记录
# user_purchases = defaultdict(set)  # user_id -> set of purchased item_ids
# item_users = defaultdict(set)      # item_id -> set of user_ids who purchased it
#
# def process_user_file(file_path):
#     line_count = 0
#     with open(file_path, 'r') as f:
#         for line in f:
#             line_count += 1
#             if line_count % 10000000 == 0:
#                 print(f"处理文件中: {file_path}, 已处理行数: {line_count}")
#             data = line.strip().split('\t')
#             if len(data) < 6:
#                 continue
#             user_id = data[0]
#             item_id = data[1]
#             behavior_type = data[2]
#             time_str = data[5]
#             if behavior_type == '4':
#                 date_str = time_str.split(' ')[0]
#                 if date_str >= '2014-11-18' and date_str <= '2014-12-18':
#                     user_purchases[user_id].add(item_id)
#                     item_users[item_id].add(user_id)
#     print(f"完成处理文件: {file_path}, 总行数: {line_count}")
#
# # 处理两个用户行为文件
# print("开始处理用户行为文件 partA ...")
# process_user_file(user_file_a)
# print("开始处理用户行为文件 partB ...")
# process_user_file(user_file_b)
#
# print("用户购买记录数量:", len(user_purchases))
# print("商品被购买记录数量:", len(item_users))
#
# # 构建共现矩阵：对于P中的每个物品i，记录与其他物品j的共现次数
# cooccurrence = defaultdict(lambda: defaultdict(int))
# for user_id, items_set in user_purchases.items():
#     items_list = list(items_set)
#     for i in items_list:
#         if i in P_items:
#             for j in items_list:
#                 if j != i:
#                     cooccurrence[i][j] += 1
#
# # 计算相似度矩阵：使用余弦相似度
# sim_dict = defaultdict(dict)   # sim_dict[i][j] = similarity between i and j for i in P
# for i in P_items:
#     if i in cooccurrence:
#         for j, co_count in cooccurrence[i].items():
#             if i in item_users and j in item_users:
#                 len_i = len(item_users[i])
#                 len_j = len(item_users[j])
#                 if len_i > 0 and len_j > 0:
#                     sim_val = co_count / math.sqrt(len_i * len_j)
#                     sim_dict[i][j] = sim_val
#
# # 构建反向映射：对于每个物品j，存储与P中物品i的相似度
# j_sim = defaultdict(list)
# for i in P_items:
#     if i in sim_dict:
#         for j, sim_val in sim_dict[i].items():
#             j_sim[j].append((i, sim_val))
#
# # 计算每个用户的推荐分数
# user_scores = defaultdict(lambda: defaultdict(float))
# for user_id, purchased_items in user_purchases.items():
#     for j in purchased_items:
#         if j in j_sim:
#             for (i, sim_val) in j_sim[j]:
#                 user_scores[user_id][i] += sim_val
#
# # 为每个用户选择Top-10物品作为推荐
# recommendations = []
# for user_id, scores in user_scores.items():
#     # 使用堆来获取分数最高的10个物品
#     top_items = heapq.nlargest(10, scores.items(), key=lambda x: x[1])
#     for item_id, score in top_items:
#         if score > 0:
#             recommendations.append((user_id, item_id))
#
# # 输出结果到文件
# with open(output_file, 'w') as f:
#     for user_id, item_id in recommendations:
#         f.write(f"{user_id}\t{item_id}\n")
#
# print("推荐结果已写入文件:", output_file)
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