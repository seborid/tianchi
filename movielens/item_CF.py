import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
import warnings

warnings.filterwarnings('ignore')


class ItemBasedCF:
    """基于物品的协同过滤推荐系统"""

    def __init__(self, n_similar_items=20, min_ratings=5, top_n=10):
        self.n_similar_items = n_similar_items  # 考虑的相似物品数量
        self.min_ratings = min_ratings  # 物品最少评分次数阈值
        self.top_n = top_n  # 推荐列表长度
        self.item_similarity_df = None  # 物品相似度矩阵
        self.train_ratings_matrix = None  # 训练集评分矩阵
        self.movies_df = None  # 电影信息
        self.train_data = None  # 训练数据
        self.test_data = None  # 测试数据

    def load_data(self, ratings_file, users_file, movies_file, sep="::"):
        """
        加载MovieLens数据集
        参数:
            ratings_file: 评分文件路径
            users_file: 用户文件路径
            movies_file: 电影文件路径
            sep: 分隔符，默认为"::"
        """
        print("开始加载数据...")

        # 定义列名
        ratings_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
        users_cols = ['user_id', 'gender', 'age', 'occupation', 'zip_code']
        movies_cols = ['movie_id', 'title', 'genres']

        try:
            # 加载数据
            ratings_df = pd.read_csv(ratings_file, sep=sep, engine='python',
                                     names=ratings_cols, encoding='latin-1')
            users_df = pd.read_csv(users_file, sep=sep, engine='python',
                                   names=users_cols, encoding='latin-1')
            movies_df = pd.read_csv(movies_file, sep=sep, engine='python',
                                    names=movies_cols, encoding='latin-1')

            # 确保ID列类型一致
            ratings_df['user_id'] = ratings_df['user_id'].astype(int)
            ratings_df['movie_id'] = ratings_df['movie_id'].astype(int)
            movies_df['movie_id'] = movies_df['movie_id'].astype(int)

            print(f"数据加载成功: {len(ratings_df)}条评分, {len(users_df)}个用户, {len(movies_df)}部电影")
            return ratings_df, users_df, movies_df

        except Exception as e:
            print(f"数据加载错误: {e}")
            return None, None, None

    def preprocess_data(self, ratings_df):
        """
        数据预处理：过滤评分较少的用户和电影
        """
        print("预处理数据...")

        # 过滤评分较少的用户
        user_rating_count = ratings_df['user_id'].value_counts()
        active_users = user_rating_count[user_rating_count >= self.min_ratings].index
        ratings_df = ratings_df[ratings_df['user_id'].isin(active_users)]

        # 过滤评分较少的电影
        movie_rating_count = ratings_df['movie_id'].value_counts()
        popular_movies = movie_rating_count[movie_rating_count >= self.min_ratings].index
        ratings_df = ratings_df[ratings_df['movie_id'].isin(popular_movies)]

        print(f"预处理后: {len(ratings_df)}条评分, {len(active_users)}个活跃用户, {len(popular_movies)}部热门电影")
        return ratings_df

    def create_ratings_matrix(self, ratings_df):
        """
        创建用户-物品评分矩阵
        """
        print("创建评分矩阵...")
        ratings_matrix = ratings_df.pivot_table(
            index='user_id', columns='movie_id', values='rating'
        )
        ratings_matrix.fillna(0, inplace=True)
        return ratings_matrix

    def calculate_item_similarity(self, ratings_matrix):
        """
        计算物品相似度矩阵（使用余弦相似度）
        """
        print("计算物品相似度矩阵...")
        start_time = time.time()

        # 转置矩阵得到物品-用户矩阵
        item_user_matrix = ratings_matrix.T

        # 计算余弦相似度
        item_similarity = cosine_similarity(item_user_matrix)

        # 转换为DataFrame
        item_similarity_df = pd.DataFrame(
            item_similarity,
            index=item_user_matrix.index,
            columns=item_user_matrix.index
        )

        print(f"相似度矩阵计算完成，耗时: {time.time() - start_time:.2f}秒")
        return item_similarity_df

    def predict_rating(self, user_id, movie_id, ratings_matrix, item_similarity_df):
        """
        预测用户对指定电影的评分
        """
        # 获取用户已评分的电影
        user_ratings = ratings_matrix.loc[user_id]
        rated_items = user_ratings[user_ratings > 0].index

        # 如果用户没有评分记录，返回0
        if len(rated_items) == 0:
            return 0

        # 获取与目标电影最相似的N个电影
        similar_items = item_similarity_df[movie_id].sort_values(ascending=False).iloc[1:self.n_similar_items + 1]

        numerator = 0
        denominator = 0

        # 计算加权评分
        for similar_item, similarity in similar_items.items():
            if similar_item in rated_items and similarity > 0:
                rating = ratings_matrix.loc[user_id, similar_item]
                numerator += similarity * rating
                denominator += abs(similarity)

        # 如果有相似的已评分物品，返回预测评分
        if denominator > 0:
            return numerator / denominator
        else:
            return 0

    def generate_recommendations(self, user_id, ratings_matrix, item_similarity_df, movies_df):
        """
        为指定用户生成推荐列表
        """
        # 获取用户已评分和未评分的电影
        user_ratings = ratings_matrix.loc[user_id]
        rated_movies = user_ratings[user_ratings > 0].index
        unrated_movies = user_ratings[user_ratings == 0].index

        predictions = []

        # 为未评分的电影预测评分
        for movie_id in unrated_movies:
            pred_rating = self.predict_rating(user_id, movie_id, ratings_matrix, item_similarity_df)
            if pred_rating > 0:  # 只考虑有预测评分的电影
                predictions.append((movie_id, pred_rating))

        # 按预测评分排序，获取Top-N推荐
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_n_predictions = predictions[:self.top_n]

        # 添加电影信息
        recommendations = []
        for movie_id, pred_rating in top_n_predictions:
            movie_info = movies_df[movies_df['movie_id'] == movie_id]
            if not movie_info.empty:
                movie_title = movie_info['title'].iloc[0]
                movie_genres = movie_info['genres'].iloc[0]
                recommendations.append({
                    'user_id': user_id,
                    'movie_id': movie_id,
                    'title': movie_title,
                    'genres': movie_genres,
                    'predicted_rating': pred_rating
                })

        return pd.DataFrame(recommendations)

    def train_test_split_by_time(self, ratings_df, test_size=0.2):
        """
        按时间划分训练集和测试集
        """
        print("按时间划分训练集和测试集...")
        ratings_sorted = ratings_df.sort_values('timestamp')
        split_idx = int(len(ratings_sorted) * (1 - test_size))

        train_data = ratings_sorted.iloc[:split_idx]
        test_data = ratings_sorted.iloc[split_idx:]

        return train_data, test_data

    def fit(self, ratings_df, movies_df):
        """
        训练模型
        """
        self.movies_df = movies_df

        # 数据预处理
        ratings_df = self.preprocess_data(ratings_df)

        # 划分训练集和测试集
        self.train_data, self.test_data = self.train_test_split_by_time(ratings_df)

        # 创建评分矩阵
        self.train_ratings_matrix = self.create_ratings_matrix(self.train_data)

        # 计算物品相似度
        self.item_similarity_df = self.calculate_item_similarity(self.train_ratings_matrix)

        print("模型训练完成")

    def evaluate_rmse_mae(self):
        """
        评估模型预测准确性 (RMSE和MAE)
        """
        if self.item_similarity_df is None:
            raise ValueError("请先训练模型")

        print("计算RMSE和MAE...")
        start_time = time.time()

        # 从测试集中抽样一部分进行评估（提高效率）
        sample_test = self.test_data.sample(min(1000, len(self.test_data)), random_state=42)

        predictions = []
        actuals = []

        for _, row in sample_test.iterrows():
            user_id, movie_id, actual_rating = row['user_id'], row['movie_id'], row['rating']

            # 确保用户和电影在训练矩阵中
            if user_id in self.train_ratings_matrix.index and movie_id in self.train_ratings_matrix.columns:
                pred_rating = self.predict_rating(user_id, movie_id, self.train_ratings_matrix, self.item_similarity_df)
                if pred_rating > 0:  # 只考虑有预测的情况
                    predictions.append(pred_rating)
                    actuals.append(actual_rating)

        # 计算评估指标
        if predictions:
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            mae = mean_absolute_error(actuals, predictions)
            eval_time = time.time() - start_time
            print(f"评估完成，耗时: {eval_time:.2f}秒")
            return rmse, mae, len(predictions)
        else:
            return None, None, 0

    def evaluate_precision_recall(self, rating_threshold=4):
        """
        评估推荐列表质量 (Precision@K和Recall@K)
        """
        if self.item_similarity_df is None:
            raise ValueError("请先训练模型")

        print("计算Precision@K和Recall@K...")
        start_time = time.time()

        # 获取测试用户
        test_users = self.test_data['user_id'].unique()
        sample_users = test_users[:min(100, len(test_users))]  # 抽样部分用户提高效率

        precision_sum = 0
        recall_sum = 0
        user_count = 0

        for user_id in sample_users:
            # 确保用户在训练集中
            if user_id not in self.train_ratings_matrix.index:
                continue

            # 获取用户在测试集中的高评分电影
            user_test_ratings = self.test_data[self.test_data['user_id'] == user_id]
            user_high_ratings = user_test_ratings[user_test_ratings['rating'] >= rating_threshold]
            high_rated_movies = user_high_ratings['movie_id'].values

            if len(high_rated_movies) == 0:
                continue

            # 为用户生成推荐
            recommendations = self.generate_recommendations(
                user_id, self.train_ratings_matrix, self.item_similarity_df, self.movies_df
            )

            if recommendations.empty:
                continue

            # 获取推荐电影ID
            recommended_movies = recommendations['movie_id'].values

            # 计算交集（推荐的相关电影）
            relevant_recommended = set(high_rated_movies) & set(recommended_movies)

            # 计算Precision和Recall
            precision = len(relevant_recommended) / len(recommended_movies) if len(recommended_movies) > 0 else 0
            recall = len(relevant_recommended) / len(high_rated_movies) if len(high_rated_movies) > 0 else 0

            precision_sum += precision
            recall_sum += recall
            user_count += 1

        # 计算平均值
        precision_avg = precision_sum / user_count if user_count > 0 else 0
        recall_avg = recall_sum / user_count if user_count > 0 else 0
        f1_score = 2 * (precision_avg * recall_avg) / (precision_avg + recall_avg) if (precision_avg + recall_avg) > 0 else 0

        eval_time = time.time() - start_time
        print(f"评估完成，耗时: {eval_time:.2f}秒")
        return precision_avg, recall_avg, f1_score, user_count

    def generate_all_recommendations(self, output_file):
        """
        为所有用户生成推荐并写入文件
        """
        if self.item_similarity_df is None:
            raise ValueError("请先训练模型")

        print("为所有用户生成推荐...")
        start_time = time.time()

        all_recommendations = []

        # 获取所有用户
        all_users = self.train_ratings_matrix.index

        for i, user_id in enumerate(all_users):
            if i % 100 == 0:
                print(f"已处理 {i}/{len(all_users)} 个用户")

            recommendations = self.generate_recommendations(
                user_id, self.train_ratings_matrix, self.item_similarity_df, self.movies_df
            )

            if not recommendations.empty:
                all_recommendations.append(recommendations)

        # 合并所有推荐结果
        if all_recommendations:
            all_recommendations_df = pd.concat(all_recommendations, ignore_index=True)
            # 写入CSV文件
            all_recommendations_df.to_csv(output_file, index=False)
            print(f"推荐结果已写入文件: {output_file}")
        else:
            print("没有生成任何推荐结果")
            all_recommendations_df = pd.DataFrame()

        print(f"推荐生成完成，耗时: {time.time() - start_time:.2f}秒")
        return all_recommendations_df


# 主函数
def main():
    """运行推荐系统"""
    # 初始化推荐系统
    rec_sys = ItemBasedCF(n_similar_items=20, min_ratings=5, top_n=10)

    # 加载数据（请替换为您的实际文件路径）
    ratings_file = "./data/ratings.dat"
    users_file = "./data/users.dat"
    movies_file = "./data/movies.dat"

    ratings_df, users_df, movies_df = rec_sys.load_data(ratings_file, users_file, movies_file, sep="::")

    if ratings_df is None:
        print("数据加载失败，请检查文件路径")
        return

    # 训练模型
    rec_sys.fit(ratings_df, movies_df)

    # 评估模型
    print("\n开始离线评估...")

    # 评估预测准确性
    rmse, mae, evaluated_count = rec_sys.evaluate_rmse_mae()
    if rmse is not None:
        print(f"RMSE (均方根误差): {rmse:.4f}")
        print(f"MAE (平均绝对误差): {mae:.4f}")
        print(f"基于 {evaluated_count} 个预测进行评估")

    # 评估推荐列表质量
    precision, recall, f1, user_count = rec_sys.evaluate_precision_recall(rating_threshold=4)
    print(f"Precision@{rec_sys.top_n}: {precision:.4f}")
    print(f"Recall@{rec_sys.top_n}: {recall:.4f}")
    print(f"F1-Score@{rec_sys.top_n}: {f1:.4f}")
    print(f"基于 {user_count} 个用户进行评估")

    # 为所有用户生成推荐并写入文件
    output_filename = "item_cf_recommendations.csv"
    recommendations_df = rec_sys.generate_all_recommendations(output_filename)

    # 显示前几个用户的推荐结果
    if not recommendations_df.empty:
        print("\n=== 前几个用户的推荐示例 ===")
        sample_users = recommendations_df['user_id'].unique()[:3]
        for user_id in sample_users:
            user_recs = recommendations_df[recommendations_df['user_id'] == user_id]
            print(f"\n用户 {user_id} 的Top-{rec_sys.top_n}推荐:")
            print(user_recs[['title', 'predicted_rating']].to_string(index=False))

    return recommendations_df


if __name__ == "__main__":
    main()
# import numpy as np
# import pandas as pd
# from scipy.sparse import csr_matrix, lil_matrix
# from sklearn.metrics.pairwise import cosine_similarity
# import time
# from tqdm import tqdm
# import os
# import gc
#
#
# class MovieLensItemCF:
#     def __init__(self):
#         self.user_item_matrix = None
#         self.item_similarity = None
#         self.user_ids = None
#         self.item_ids = None
#         self.user_index_map = None
#         self.item_index_map = None
#
#     def load_data(self, ratings_file, movies_file=None, users_file=None):
#         """
#         加载MovieLens数据集
#         """
#         print("开始加载评分数据...")
#         start_time = time.time()
#
#         # 读取评分数据
#         ratings = pd.read_csv(
#             ratings_file,
#             sep='::',
#             engine='python',
#             names=['user_id', 'movie_id', 'rating', 'timestamp']
#         )
#
#         # 读取电影数据（可选）
#         if movies_file and os.path.exists(movies_file):
#             movies = pd.read_csv(
#                 movies_file,
#                 sep='::',
#                 engine='python',
#                 names=['movie_id', 'title', 'genres'],
#                 encoding='latin-1'
#             )
#             # 将电影标题添加到评分数据中
#             ratings = ratings.merge(movies[['movie_id', 'title']], on='movie_id', how='left')
#
#         print(f"评分数据加载完成，共 {len(ratings)} 条评分记录，耗时 {time.time() - start_time:.2f} 秒")
#
#         # 创建用户和电影的索引映射
#         self.user_ids = sorted(ratings['user_id'].unique())
#         self.item_ids = sorted(ratings['movie_id'].unique())
#         self.user_index_map = {user: idx for idx, user in enumerate(self.user_ids)}
#         self.item_index_map = {item: idx for idx, item in enumerate(self.item_ids)}
#
#         print(f"共有 {len(self.user_ids)} 个用户，{len(self.item_ids)} 部电影")
#
#         # 构建用户-电影评分矩阵
#         rows = [self.user_index_map[user] for user in ratings['user_id']]
#         cols = [self.item_index_map[movie] for movie in ratings['movie_id']]
#         data = ratings['rating'].values
#
#         self.user_item_matrix = csr_matrix(
#             (data, (rows, cols)),
#             shape=(len(self.user_ids), len(self.item_ids))
#         )
#
#         return ratings
#
#     def calculate_similarity(self, min_ratings=5, top_k=50):
#         """
#         计算电影之间的余弦相似度
#         """
#         print("开始计算电影相似度...")
#         start_time = time.time()
#
#         # 计算每部电影的评分次数
#         movie_ratings = np.array(self.user_item_matrix.astype(bool).sum(axis=0)).flatten()
#
#         # 过滤评分次数过少的电影
#         valid_movies = np.where(movie_ratings >= min_ratings)[0]
#         print(f"有效电影数量: {len(valid_movies)} (评分次数 >= {min_ratings})")
#
#         if len(valid_movies) == 0:
#             print("没有足够的有效电影计算相似度")
#             return False
#
#         # 只处理有效电影
#         filtered_matrix = self.user_item_matrix[:, valid_movies]
#
#         # 计算余弦相似度
#         print("计算电影相似度矩阵...")
#         similarity_matrix = cosine_similarity(filtered_matrix.T, dense_output=False)
#
#         # 只保留Top-K最相似的电影
#         print("保留Top-K相似电影...")
#         self.item_similarity = lil_matrix((len(self.item_ids), len(self.item_ids)))
#
#         for i in tqdm(range(similarity_matrix.shape[0]), desc="处理电影相似度"):
#             row = similarity_matrix.getrow(i).toarray().flatten()
#             # 获取Top-K最相似的电影（不包括自身）
#             top_indices = np.argpartition(row, -top_k - 1)[-top_k - 1:-1]
#             for j in top_indices:
#                 if i != j and row[j] > 0:
#                     orig_i = valid_movies[i]
#                     orig_j = valid_movies[j]
#                     self.item_similarity[orig_i, orig_j] = row[j]
#
#         # 转换为CSR格式以提高后续操作效率
#         self.item_similarity = self.item_similarity.tocsr()
#         print(f"相似度计算完成，耗时 {time.time() - start_time:.2f} 秒")
#         return True
#
#     def predict_rating(self, user_idx, item_idx):
#         """
#         预测用户对电影的评分
#         """
#         user_ratings = self.user_item_matrix[user_idx, :].toarray().flatten()
#
#         # 获取用户已评分的电影索引
#         rated_indices = np.where(user_ratings > 0)[0]
#
#         if len(rated_indices) == 0:
#             return 0
#
#         # 获取当前电影的相似电影
#         similarities = self.item_similarity[item_idx, :].toarray().flatten()
#
#         # 只考虑用户已评分的电影
#         rated_similarities = similarities[rated_indices]
#         rated_values = user_ratings[rated_indices]
#
#         # 计算加权平均评分
#         if np.sum(np.abs(rated_similarities)) > 0:
#             predicted_rating = np.dot(rated_similarities, rated_values) / np.sum(np.abs(rated_similarities))
#         else:
#             predicted_rating = 0
#
#         return predicted_rating
#
#     def generate_recommendations(self, user_id, top_n=1):
#         """
#         为指定用户生成推荐
#         """
#         if user_id not in self.user_index_map:
#             return []
#
#         user_idx = self.user_index_map[user_id]
#         user_ratings = self.user_item_matrix[user_idx, :].toarray().flatten()
#
#         # 获取用户未评分的电影
#         unrated_indices = np.where(user_ratings == 0)[0]
#
#         if len(unrated_indices) == 0:
#             return []
#
#         # 预测用户对未评分电影的评分
#         predictions = []
#         for item_idx in unrated_indices:
#             predicted_rating = self.predict_rating(user_idx, item_idx)
#             if predicted_rating > 0:
#                 predictions.append((self.item_ids[item_idx], predicted_rating))
#
#         # 按预测评分排序
#         predictions.sort(key=lambda x: x[1], reverse=True)
#
#         # 返回Top-N推荐
#         return [movie_id for movie_id, _ in predictions[:top_n]]
#
#     def generate_submission(self, ratings_file, output_file, top_n=1):
#         """
#         生成提交文件
#         """
#         # 加载数据
#         ratings = self.load_data(ratings_file)
#
#         # 计算相似度
#         if not self.calculate_similarity(min_ratings=5, top_k=50):
#             print("相似度计算失败")
#             return None
#
#         # 生成推荐结果
#         results = []
#         print("生成推荐结果...")
#         start_time = time.time()
#
#         for user_id in tqdm(self.user_ids, desc="为用户生成推荐"):
#             recommendations = self.generate_recommendations(user_id, top_n=top_n)
#             for movie_id in recommendations:
#                 results.append((user_id, movie_id))
#
#         # 保存结果
#         with open(output_file, 'w', encoding='utf-8') as f:
#             for user_id, movie_id in results:
#                 f.write(f"{user_id}\t{movie_id}\n")
#
#         print(f"生成推荐结果完成，共{len(results)}条推荐记录，耗时 {time.time() - start_time:.2f} 秒")
#         return results
#
#
# # 主程序
# if __name__ == "__main__":
#     # 初始化推荐系统
#     recommender = MovieLensItemCF()
#
#     # 文件路径设置
#     data_dir = "data"  # 数据目录
#     ratings_file = os.path.join(data_dir, "ratings.dat")
#     movies_file = os.path.join(data_dir, "movies.dat")
#     output_file = os.path.join(data_dir, "movie_recommendations.txt")
#
#     # 检查文件是否存在
#     if not os.path.exists(ratings_file):
#         print(f"错误: 文件 {ratings_file} 不存在")
#         exit(1)
#
#     # 生成推荐结果
#     results = recommender.generate_submission(ratings_file, output_file, top_n=1)
#
#     if results is not None:
#         # 打印前10条推荐结果
#         print("\n推荐结果示例:")
#         for i in range(min(10, len(results))):
#             print(f"{results[i][0]}\t{results[i][1]}")
#     else:
#         print("推荐结果生成失败")