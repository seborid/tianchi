# 文件路径配置
DATA_DIR = "/tianchi/data/"
USER_DATA_PATHS = [
    DATA_DIR + "tianchi_fresh_comp_train_user_online_partA.txt",
    DATA_DIR + "tianchi_fresh_comp_train_user_online_partB.txt"
]
ITEM_DATA_PATH = DATA_DIR + "tianchi_fresh_comp_train_item_online.txt"
OUTPUT_PATH = DATA_DIR + "item_cf_recommendations.txt"

# 算法参数配置
CHUNK_SIZE = 100000000  # 每次读取的数据块大小
BEHAVIOR_WEIGHTS = {
    '1': 1,  # 浏览
    '2': 2,  # 收藏
    '3': 3,  # 加购物车
    '4': 4   # 购买
}
SIMILARITY_THRESHOLD = 0.1  # 相似度阈值
TOP_N_RECOMMENDATIONS = 1  # 为每个用户推荐的商品数量

# 内存优化配置
MAX_ITEMS_IN_MEMORY = 10000  # 内存中最多保留的物品数量
USE_SPARSE_MATRIX = True     # 是否使用稀疏矩阵存储