import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
import time
import warnings

warnings.filterwarnings('ignore')


class UserBasedCFRecommender:
    """基于用户的协同过滤电影推荐系统"""

    def __init__(self, n_similar_users=20, min_ratings=5, top_n=10):
        self.n_similar_users = n_similar_users
        self.min_ratings = min_ratings
        self.top_n = top_n
        self.user_similarity_df = None
        self.user_movie_ratings = None
        self.train_ratings = None
        self.movies_df = None

    def load_data(self, ratings_file, users_file, movies_file):
        """加载MovieLens数据集"""
        print("加载数据...")
        try:
            # 定义列名
            ratings_cols = ['UserID', 'MovieID', 'Rating', 'Timestamp']
            users_cols = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code']
            movies_cols = ['MovieID', 'Title', 'Genres']

            # 加载数据
            ratings_df = pd.read_csv(ratings_file, sep='::', engine='python',
                                     names=ratings_cols, encoding='latin-1')
            users_df = pd.read_csv(users_file, sep='::', engine='python',
                                   names=users_cols, encoding='latin-1')
            movies_df = pd.read_csv(movies_file, sep='::', engine='python',
                                    names=movies_cols, encoding='latin-1')

            # 确保MovieID列类型一致
            ratings_df['MovieID'] = ratings_df['MovieID'].astype(int)
            movies_df['MovieID'] = movies_df['MovieID'].astype(int)

            return ratings_df, users_df, movies_df
        except Exception as e:
            print(f"数据加载错误: {e}")
            return None, None, None

    def preprocess_data(self, ratings_df):
        """数据预处理"""
        print("预处理数据...")
        # 过滤评分较少的用户和电影
        user_rating_count = ratings_df['UserID'].value_counts()
        active_users = user_rating_count[user_rating_count >= self.min_ratings].index
        ratings_df = ratings_df[ratings_df['UserID'].isin(active_users)]

        movie_rating_count = ratings_df['MovieID'].value_counts()
        popular_movies = movie_rating_count[movie_rating_count >= self.min_ratings].index
        ratings_df = ratings_df[ratings_df['MovieID'].isin(popular_movies)]

        return ratings_df

    def create_user_movie_matrix(self, ratings_df):
        """创建用户-电影评分矩阵"""
        print("创建用户-电影评分矩阵...")
        user_movie_ratings = ratings_df.pivot_table(
            index='UserID', columns='MovieID', values='Rating'
        )
        user_movie_ratings.fillna(0, inplace=True)
        return user_movie_ratings

    def calculate_user_similarity(self, user_movie_ratings):
        """计算用户相似度矩阵"""
        print("计算用户相似度矩阵...")
        start_time = time.time()

        # 使用稀疏矩阵提高计算效率
        sparse_matrix = csr_matrix(user_movie_ratings.values)

        # 计算余弦相似度
        user_similarity = cosine_similarity(sparse_matrix)

        # 转换为DataFrame
        user_similarity_df = pd.DataFrame(
            user_similarity,
            index=user_movie_ratings.index,
            columns=user_movie_ratings.index
        )

        print(f"相似度矩阵计算完成，耗时: {time.time() - start_time:.2f}秒")
        return user_similarity_df

    def predict_ratings(self, user_id, user_movie_ratings, user_similarity_df):
        """为用户预测对未评分电影的评分"""
        # 获取目标用户已评分和未评分的电影
        target_user_ratings = user_movie_ratings.loc[user_id]
        unrated_movies = target_user_ratings[target_user_ratings == 0].index

        # 获取最相似的N个用户（排除自己）
        similar_users = user_similarity_df[user_id].sort_values(ascending=False).iloc[1:self.n_similar_users + 1].index

        # 计算预测评分
        predictions = {}
        for movie_id in unrated_movies:
            weighted_sum = 0
            similarity_sum = 0

            for sim_user_id in similar_users:
                if user_movie_ratings.loc[sim_user_id, movie_id] > 0:
                    similarity = user_similarity_df.loc[user_id, sim_user_id]
                    weighted_sum += similarity * user_movie_ratings.loc[sim_user_id, movie_id]
                    similarity_sum += similarity

            if similarity_sum > 0:
                predictions[movie_id] = weighted_sum / similarity_sum

        return predictions

    def get_top_n_recommendations(self, user_id, user_movie_ratings, user_similarity_df):
        """获取Top-N推荐"""
        predictions = self.predict_ratings(user_id, user_movie_ratings, user_similarity_df)

        # 按预测评分排序
        recommended_movies = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:self.top_n]

        # 添加电影信息
        recommendations = []
        for movie_id, pred_rating in recommended_movies:
            # 确保电影ID在movies_df中存在
            if movie_id in self.movies_df['MovieID'].values:
                movie_info = self.movies_df[self.movies_df['MovieID'] == movie_id].iloc[0]
                recommendations.append({
                    'UserID': user_id,
                    'MovieID': movie_id,
                    'Title': movie_info['Title'],
                    'Genres': movie_info['Genres'],
                    'Predicted_Rating': round(pred_rating, 2)
                })

        return pd.DataFrame(recommendations)

    def cold_start_recommendation(self, user_id):
        """冷启动推荐策略"""
        # 返回最受欢迎的电影
        movie_ratings_count = self.train_ratings['MovieID'].value_counts()
        popular_movies = movie_ratings_count.head(self.top_n).index

        recommendations = []
        for movie_id in popular_movies:
            if movie_id in self.movies_df['MovieID'].values:
                movie_info = self.movies_df[self.movies_df['MovieID'] == movie_id].iloc[0]
                recommendations.append({
                    'UserID': user_id,
                    'MovieID': movie_id,
                    'Title': movie_info['Title'],
                    'Genres': movie_info['Genres'],
                    'Predicted_Rating': 4.0  # 默认评分
                })

        return pd.DataFrame(recommendations)

    def train_test_split_by_time(self, ratings_df, test_size=0.2):
        """按时间划分训练集和测试集"""
        print("按时间划分训练集和测试集...")
        # 按时间戳排序
        ratings_sorted = ratings_df.sort_values('Timestamp')

        # 计算分割点
        split_idx = int(len(ratings_sorted) * (1 - test_size))

        # 划分数据集
        train_ratings = ratings_sorted.iloc[:split_idx]
        test_ratings = ratings_sorted.iloc[split_idx:]

        return train_ratings, test_ratings

    def fit(self, ratings_df, movies_df):
        """训练模型"""
        # 保存movies_df供后续使用
        self.movies_df = movies_df

        # 预处理数据
        ratings_df = self.preprocess_data(ratings_df)

        # 划分训练集和测试集
        self.train_ratings, self.test_ratings = self.train_test_split_by_time(ratings_df)

        # 创建用户-电影评分矩阵
        self.user_movie_ratings = self.create_user_movie_matrix(self.train_ratings)

        # 计算用户相似度
        self.user_similarity_df = self.calculate_user_similarity(self.user_movie_ratings)

        print("模型训练完成")

    def generate_recommendations_for_all_users(self, output_file):
        """为所有用户生成推荐并写入文件"""
        if self.user_similarity_df is None:
            raise ValueError("请先训练模型")

        print("开始为所有用户生成推荐...")
        start_time = time.time()

        # 获取所有用户ID（包括训练集和测试集中的用户）
        all_users = pd.concat([self.train_ratings, self.test_ratings])['UserID'].unique()

        all_recommendations = []

        # 为每个用户生成推荐
        for i, user_id in enumerate(all_users):
            if i % 100 == 0:
                print(f"已处理 {i}/{len(all_users)} 个用户")

            if user_id in self.user_movie_ratings.index:
                # 用户存在于训练集中，使用协同过滤
                print(f"为用户 {user_id} 生成推荐...")
                recommendations = self.get_top_n_recommendations(
                    user_id, self.user_movie_ratings, self.user_similarity_df
                )
            else:
                # 用户不存在于训练集中，使用冷启动策略
                print(f"用户 {user_id} 不在训练集中，使用冷启动推荐...")
                recommendations = self.cold_start_recommendation(user_id)

            all_recommendations.append(recommendations)

        # 合并所有推荐结果
        all_recommendations_df = pd.concat(all_recommendations, ignore_index=True)

        # 写入文件
        all_recommendations_df.to_csv(output_file, index=False)

        print(f"推荐生成完成，耗时: {time.time() - start_time:.2f}秒")
        print(f"推荐结果已保存到: {output_file}")

        return all_recommendations_df


# 主函数
def main():
    """运行推荐系统"""
    # 初始化推荐系统
    rec_sys = UserBasedCFRecommender(n_similar_users=15, min_ratings=5, top_n=10)

    # 加载数据
    ratings_df, users_df, movies_df = rec_sys.load_data(
        './data/ratings.dat', './data/users.dat', './data/movies.dat'
    )

    if ratings_df is None:
        print("数据加载失败，请检查文件路径和格式")
        return

    # 训练模型
    rec_sys.fit(ratings_df, movies_df)

    # 为所有用户生成推荐并写入文件
    recommendations_df = rec_sys.generate_recommendations_for_all_users(
        'user_recommendations.csv'
    )

    # 显示前几个用户的推荐结果
    print("\n前几个用户的推荐结果:")
    sample_users = recommendations_df['UserID'].unique()[:3]
    for user_id in sample_users:
        user_recommendations = recommendations_df[recommendations_df['UserID'] == user_id]
        print(f"\n用户 {user_id} 的推荐:")
        print(user_recommendations[['Title', 'Genres', 'Predicted_Rating']].to_string(index=False))

    return recommendations_df


if __name__ == "__main__":
    main()
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# from scipy.sparse import csr_matrix
# import time
# import warnings
#
# warnings.filterwarnings('ignore')
#
#
# class UserBasedCF:
#     """基于用户的协同过滤推荐系统"""
#
#     def __init__(self, n_similar_users=20, min_ratings=5):
#         self.n_similar_users = n_similar_users
#         self.min_ratings = min_ratings
#         self.user_similarity_df = None
#         self.user_movie_ratings = None
#         self.train_ratings = None
#         self.movies_df = None
#
#     def load_data(self, ratings_file, users_file, movies_file):
#         """加载MovieLens数据集"""
#         print("加载数据...")
#         # 定义列名
#         ratings_cols = ['UserID', 'MovieID', 'Rating', 'Timestamp']
#         users_cols = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code']
#         movies_cols = ['MovieID', 'Title', 'Genres']
#
#         try:
#             # 加载数据
#             ratings_df = pd.read_csv(ratings_file, sep='::', engine='python',
#                                      names=ratings_cols, encoding='latin-1')
#             users_df = pd.read_csv(users_file, sep='::', engine='python',
#                                    names=users_cols, encoding='latin-1')
#             movies_df = pd.read_csv(movies_file, sep='::', engine='python',
#                                     names=movies_cols, encoding='latin-1')
#
#             # 确保MovieID列类型一致
#             ratings_df['MovieID'] = ratings_df['MovieID'].astype(int)
#             movies_df['MovieID'] = movies_df['MovieID'].astype(int)
#
#             return ratings_df, users_df, movies_df
#         except Exception as e:
#             print(f"数据加载错误: {e}")
#             return None, None, None
#
#     def preprocess_data(self, ratings_df):
#         """数据预处理"""
#         print("预处理数据...")
#         # 过滤评分较少的用户和电影
#         user_rating_count = ratings_df['UserID'].value_counts()
#         active_users = user_rating_count[user_rating_count >= self.min_ratings].index
#         ratings_df = ratings_df[ratings_df['UserID'].isin(active_users)]
#
#         movie_rating_count = ratings_df['MovieID'].value_counts()
#         popular_movies = movie_rating_count[movie_rating_count >= self.min_ratings].index
#         ratings_df = ratings_df[ratings_df['MovieID'].isin(popular_movies)]
#
#         return ratings_df
#
#     def create_user_movie_matrix(self, ratings_df):
#         """创建用户-电影评分矩阵"""
#         print("创建用户-电影评分矩阵...")
#         user_movie_ratings = ratings_df.pivot_table(
#             index='UserID', columns='MovieID', values='Rating'
#         )
#         user_movie_ratings.fillna(0, inplace=True)
#         return user_movie_ratings
#
#     def calculate_user_similarity(self, user_movie_ratings):
#         """计算用户相似度矩阵"""
#         print("计算用户相似度矩阵...")
#         start_time = time.time()
#
#         # 使用稀疏矩阵提高计算效率
#         sparse_matrix = csr_matrix(user_movie_ratings.values)
#
#         # 计算余弦相似度
#         from sklearn.metrics.pairwise import cosine_similarity
#         user_similarity = cosine_similarity(sparse_matrix)
#
#         # 转换为DataFrame
#         user_similarity_df = pd.DataFrame(
#             user_similarity,
#             index=user_movie_ratings.index,
#             columns=user_movie_ratings.index
#         )
#
#         print(f"相似度矩阵计算完成，耗时: {time.time() - start_time:.2f}秒")
#         return user_similarity_df
#
#     def predict_ratings(self, user_id, user_movie_ratings, user_similarity_df):
#         """为用户预测对未评分电影的评分"""
#         # 获取目标用户已评分和未评分的电影
#         target_user_ratings = user_movie_ratings.loc[user_id]
#         unrated_movies = target_user_ratings[target_user_ratings == 0].index
#
#         # 获取最相似的N个用户（排除自己）
#         similar_users = user_similarity_df[user_id].sort_values(ascending=False).iloc[1:self.n_similar_users + 1].index
#
#         # 计算预测评分
#         predictions = {}
#         for movie_id in unrated_movies:
#             weighted_sum = 0
#             similarity_sum = 0
#
#             for sim_user_id in similar_users:
#                 if user_movie_ratings.loc[sim_user_id, movie_id] > 0:
#                     similarity = user_similarity_df.loc[user_id, sim_user_id]
#                     weighted_sum += similarity * user_movie_ratings.loc[sim_user_id, movie_id]
#                     similarity_sum += similarity
#
#             if similarity_sum > 0:
#                 predictions[movie_id] = weighted_sum / similarity_sum
#
#         return predictions
#
#     def get_top_n_recommendations(self, user_id, user_movie_ratings, user_similarity_df, top_n=10):
#         """获取Top-N推荐"""
#         predictions = self.predict_ratings(user_id, user_movie_ratings, user_similarity_df)
#
#         # 按预测评分排序
#         recommended_movies = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:top_n]
#
#         # 添加电影信息
#         recommendations = []
#         for movie_id, pred_rating in recommended_movies:
#             # 确保电影ID在movies_df中存在
#             if movie_id in self.movies_df['MovieID'].values:
#                 movie_info = self.movies_df[self.movies_df['MovieID'] == movie_id].iloc[0]
#                 recommendations.append({
#                     'MovieID': movie_id,
#                     'Title': movie_info['Title'],
#                     'Genres': movie_info['Genres'],
#                     'Predicted_Rating': round(pred_rating, 2)
#                 })
#
#         return pd.DataFrame(recommendations)
#
#     def evaluate_rmse_mae(self, test_ratings, user_movie_ratings, user_similarity_df):
#         """评估RMSE和MAE"""
#         print("计算RMSE和MAE...")
#         all_preds = []
#         all_actuals = []
#
#         # 只评估部分测试数据以提高速度
#         sample_test = test_ratings.sample(min(1000, len(test_ratings)), random_state=42)
#
#         for user_id, movie_id, actual_rating in sample_test[['UserID', 'MovieID', 'Rating']].values:
#             if user_id in user_similarity_df.index and movie_id in user_movie_ratings.columns:
#                 predictions = self.predict_ratings(user_id, user_movie_ratings, user_similarity_df)
#                 pred_rating = predictions.get(movie_id, 0)
#
#                 if pred_rating > 0:
#                     all_preds.append(pred_rating)
#                     all_actuals.append(actual_rating)
#
#         if all_preds:
#             rmse = np.sqrt(mean_squared_error(all_actuals, all_preds))
#             mae = mean_absolute_error(all_actuals, all_preds)
#             return rmse, mae, len(all_preds)
#         else:
#             return 0, 0, 0
#
#     def evaluate_precision_recall(self, test_ratings, user_movie_ratings, user_similarity_df, top_n=10):
#         """评估Precision和Recall"""
#         print("计算Precision和Recall...")
#         precision_sum = 0
#         recall_sum = 0
#         user_count = 0
#
#         # 只评估部分用户以提高速度
#         test_users = test_ratings['UserID'].unique()[:100]
#
#         for user_id in test_users:
#             if user_id not in user_similarity_df.index:
#                 continue
#
#             # 获取用户的测试评分
#             user_test_ratings = test_ratings[test_ratings['UserID'] == user_id]
#             user_high_ratings = user_test_ratings[user_test_ratings['Rating'] >= 4]['MovieID'].values
#
#             if len(user_high_ratings) == 0:
#                 continue
#
#             # 获取推荐
#             recommendations = self.get_top_n_recommendations(
#                 user_id, user_movie_ratings, user_similarity_df, top_n
#             )
#
#             if recommendations.empty:
#                 continue
#
#             recommended_movies = recommendations['MovieID'].values
#
#             # 计算相关项目
#             relevant_items = set(user_high_ratings)
#             recommended_items = set(recommended_movies)
#
#             # 计算Precision和Recall
#             if len(recommended_items) > 0:
#                 precision = len(relevant_items.intersection(recommended_items)) / len(recommended_items)
#                 recall = len(relevant_items.intersection(recommended_items)) / len(relevant_items)
#
#                 precision_sum += precision
#                 recall_sum += recall
#                 user_count += 1
#
#         # 计算平均值
#         precision_at_k = precision_sum / user_count if user_count > 0 else 0
#         recall_at_k = recall_sum / user_count if user_count > 0 else 0
#         f1_score = 2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k) if (
#                                                                                                       precision_at_k + recall_at_k) > 0 else 0
#
#         return precision_at_k, recall_at_k, f1_score, user_count
#
#     def train_test_split_by_time(self, ratings_df, test_size=0.2):
#         """按时间划分训练集和测试集"""
#         print("按时间划分训练集和测试集...")
#         # 按时间戳排序
#         ratings_sorted = ratings_df.sort_values('Timestamp')
#
#         # 计算分割点
#         split_idx = int(len(ratings_sorted) * (1 - test_size))
#
#         # 划分数据集
#         train_ratings = ratings_sorted.iloc[:split_idx]
#         test_ratings = ratings_sorted.iloc[split_idx:]
#
#         return train_ratings, test_ratings
#
#     def fit(self, ratings_df, movies_df):
#         """训练模型"""
#         # 保存movies_df供后续使用
#         self.movies_df = movies_df
#
#         # 预处理数据
#         ratings_df = self.preprocess_data(ratings_df)
#
#         # 划分训练集和测试集
#         self.train_ratings, self.test_ratings = self.train_test_split_by_time(ratings_df)
#
#         # 创建用户-电影评分矩阵
#         self.user_movie_ratings = self.create_user_movie_matrix(self.train_ratings)
#
#         # 计算用户相似度
#         self.user_similarity_df = self.calculate_user_similarity(self.user_movie_ratings)
#
#         print("模型训练完成")
#
#     def evaluate(self):
#         """评估模型性能"""
#         if self.user_similarity_df is None:
#             raise ValueError("请先训练模型")
#
#         print("开始评估模型...")
#         start_time = time.time()
#
#         # 评估RMSE和MAE
#         rmse, mae, evaluated_count = self.evaluate_rmse_mae(
#             self.test_ratings, self.user_movie_ratings, self.user_similarity_df
#         )
#
#         # 评估Precision和Recall
#         precision, recall, f1, user_count = self.evaluate_precision_recall(
#             self.test_ratings, self.user_movie_ratings, self.user_similarity_df
#         )
#
#         print(f"评估完成，耗时: {time.time() - start_time:.2f}秒")
#         print("\n评估结果:")
#         print(f"RMSE: {rmse:.4f} (基于{evaluated_count}个预测)")
#         print(f"MAE: {mae:.4f} (基于{evaluated_count}个预测)")
#         print(f"Precision@10: {precision:.4f} (基于{user_count}个用户)")
#         print(f"Recall@10: {recall:.4f} (基于{user_count}个用户)")
#         print(f"F1-Score@10: {f1:.4f} (基于{user_count}个用户)")
#
#         return {
#             'RMSE': rmse,
#             'MAE': mae,
#             'Precision@10': precision,
#             'Recall@10': recall,
#             'F1-Score@10': f1,
#             'Evaluated_Users': user_count,
#             'Evaluated_Ratings': evaluated_count
#         }
#
#     def recommend_for_user(self, user_id, top_n=10):
#         """为用户生成推荐"""
#         if self.user_similarity_df is None:
#             raise ValueError("请先训练模型")
#
#         if user_id not in self.user_movie_ratings.index:
#             print(f"用户 {user_id} 不在训练集中")
#             return pd.DataFrame(columns=['MovieID', 'Title', 'Genres', 'Predicted_Rating'])
#
#         recommendations = self.get_top_n_recommendations(
#             user_id, self.user_movie_ratings, self.user_similarity_df, top_n
#         )
#
#         return recommendations
#
#
# # 主函数
# def main():
#     """运行推荐系统"""
#     # 初始化推荐系统
#     rec_sys = UserBasedCF(n_similar_users=15, min_ratings=5)
#
#     # 加载数据
#     ratings_df, users_df, movies_df = rec_sys.load_data(
#         './data/ratings.dat', './data/users.dat', './data/movies.dat'
#     )
#
#     if ratings_df is None:
#         print("数据加载失败，请检查文件路径和格式")
#         return
#
#     # 训练模型
#     rec_sys.fit(ratings_df, movies_df)
#
#     # 评估模型
#     evaluation_results = rec_sys.evaluate()
#
#     # 为示例用户生成推荐
#     example_user_id = ratings_df['UserID'].iloc[0]
#     print(f"\n为用户 {example_user_id} 生成推荐:")
#     recommendations = rec_sys.recommend_for_user(example_user_id)
#
#     # 安全地打印推荐结果
#     if recommendations.empty:
#         print("没有推荐结果")
#     else:
#         # 检查所需的列是否存在
#         required_columns = ['Title', 'Genres', 'Predicted_Rating']
#         if all(col in recommendations.columns for col in required_columns):
#             print(recommendations[required_columns])
#         else:
#             print("推荐结果中缺少某些列")
#             print("实际列名:", recommendations.columns.tolist())
#             print("完整推荐结果:")
#             print(recommendations)
#
#     return evaluation_results, recommendations
#
#
# if __name__ == "__main__":
#     main()
# import pandas as pd
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# import warnings
#
# warnings.filterwarnings('ignore')
#
#
# # 1. 数据加载函数
# def load_movielens_data():
#     """加载MovieLens数据集"""
#     # 定义列名
#     ratings_cols = ['UserID', 'MovieID', 'Rating', 'Timestamp']
#     users_cols = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code']
#     movies_cols = ['MovieID', 'Title', 'Genres']
#
#     # 加载数据
#     ratings_df = pd.read_csv('./data/ratings.dat', sep='::', engine='python',
#                              names=ratings_cols, encoding='latin-1')
#     users_df = pd.read_csv('./data/users.dat', sep='::', engine='python',
#                            names=users_cols, encoding='latin-1')
#     movies_df = pd.read_csv('./data/movies.dat', sep='::', engine='python',
#                             names=movies_cols, encoding='latin-1')
#
#     return ratings_df, users_df, movies_df
#
#
# # 2. 数据预处理
# def preprocess_data(ratings_df, users_df, movies_df, min_ratings=5):
#     """数据预处理"""
#     # 过滤评分较少的用户
#     user_rating_count = ratings_df['UserID'].value_counts()
#     active_users = user_rating_count[user_rating_count >= min_ratings].index
#     ratings_df = ratings_df[ratings_df['UserID'].isin(active_users)]
#
#     # 过滤评分较少的电影
#     movie_rating_count = ratings_df['MovieID'].value_counts()
#     popular_movies = movie_rating_count[movie_rating_count >= min_ratings].index
#     ratings_df = ratings_df[ratings_df['MovieID'].isin(popular_movies)]
#
#     # 创建用户-电影评分矩阵
#     user_movie_ratings = ratings_df.pivot_table(
#         index='UserID', columns='MovieID', values='Rating'
#     )
#
#     # 填充缺失值为0
#     user_movie_ratings.fillna(0, inplace=True)
#
#     return ratings_df, user_movie_ratings
#
#
# # 3. 计算用户相似度
# def calculate_user_similarity(user_movie_ratings):
#     """计算用户相似度矩阵"""
#     # 使用余弦相似度
#     user_similarity = cosine_similarity(user_movie_ratings)
#     user_similarity_df = pd.DataFrame(
#         user_similarity,
#         index=user_movie_ratings.index,
#         columns=user_movie_ratings.index
#     )
#     return user_similarity_df
#
#
# # 4. 生成推荐
# def predict_ratings(user_id, user_movie_ratings, user_similarity_df, n_similar_users=20):
#     """为用户预测对未评分电影的评分"""
#     # 获取目标用户已评分和未评分的电影
#     target_user_ratings = user_movie_ratings.loc[user_id]
#     rated_movies = target_user_ratings[target_user_ratings > 0].index
#     unrated_movies = target_user_ratings[target_user_ratings == 0].index
#
#     # 获取最相似的N个用户（排除自己）
#     similar_users = user_similarity_df[user_id].sort_values(ascending=False).iloc[1:n_similar_users + 1].index
#
#     # 计算预测评分
#     predictions = {}
#     for movie_id in unrated_movies:
#         weighted_sum = 0
#         similarity_sum = 0
#
#         for sim_user_id in similar_users:
#             if user_movie_ratings.loc[sim_user_id, movie_id] > 0:
#                 similarity = user_similarity_df.loc[user_id, sim_user_id]
#                 weighted_sum += similarity * user_movie_ratings.loc[sim_user_id, movie_id]
#                 similarity_sum += similarity
#
#         if similarity_sum > 0:
#             predictions[movie_id] = weighted_sum / similarity_sum
#
#     return predictions
#
#
# def get_top_n_recommendations(user_id, user_movie_ratings, user_similarity_df, movies_df, n_similar_users=20, top_n=10):
#     """获取Top-N推荐"""
#     # 获取预测评分
#     predictions = predict_ratings(user_id, user_movie_ratings, user_similarity_df, n_similar_users)
#
#     # 按预测评分排序
#     recommended_movies = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:top_n]
#
#     # 添加电影信息
#     recommendations = []
#     for movie_id, pred_rating in recommended_movies:
#         movie_info = movies_df[movies_df['MovieID'] == movie_id].iloc[0]
#         recommendations.append({
#             'MovieID': movie_id,
#             'Title': movie_info['Title'],
#             'Genres': movie_info['Genres'],
#             'Predicted_Rating': round(pred_rating, 2)
#         })
#
#     return pd.DataFrame(recommendations)
#
#
# # 5. 评估指标
# def evaluate_recommendations(test_ratings, user_movie_ratings, user_similarity_df, movies_df, n_similar_users=20,
#                              top_n=10):
#     """评估推荐系统性能（带详细调试信息）"""
#     print("=" * 50)
#     print("开始评估推荐系统性能")
#     print("=" * 50)
#
#     # 计算RMSE和MAE
#     all_preds = []
#     all_actuals = []
#
#     # 预先过滤测试集，只保留在训练集中存在的用户和电影
#     valid_users = set(user_similarity_df.index)
#     valid_movies = set(user_movie_ratings.columns)
#
#     filtered_test = test_ratings[
#         (test_ratings['UserID'].isin(valid_users)) &
#         (test_ratings['MovieID'].isin(valid_movies))
#         ]
#
#     print(f"原始测试集大小: {len(test_ratings)} 条评分记录")
#     print(f"过滤后测试集大小: {len(filtered_test)} 条评分记录")
#     print(f"有效用户数: {len(valid_users)}")
#     print(f"有效电影数: {len(valid_movies)}")
#
#     # 分批处理，避免内存溢出
#     batch_size = 1000
#     total_batches = (len(filtered_test) - 1) // batch_size + 1
#
#     print(f"\n开始计算RMSE和MAE，共 {total_batches} 个批次")
#
#     for i in range(0, len(filtered_test), batch_size):
#         batch = filtered_test.iloc[i:i + batch_size]
#         batch_num = i // batch_size + 1
#         print(f"处理批次 {batch_num}/{total_batches}，大小: {len(batch)}")
#
#         for idx, (user_id, movie_id, actual_rating) in enumerate(batch[['UserID', 'MovieID', 'Rating']].values):
#             if idx % 200 == 0 and idx > 0:
#                 print(f"  批次内已处理 {idx}/{len(batch)} 条记录")
#
#             try:
#                 # 获取预测评分
#                 predictions = predict_ratings(user_id, user_movie_ratings, user_similarity_df, n_similar_users)
#                 pred_rating = predictions.get(movie_id, 0)
#
#                 if pred_rating > 0:
#                     all_preds.append(pred_rating)
#                     all_actuals.append(actual_rating)
#             except Exception as e:
#                 print(f"处理用户 {user_id} 电影 {movie_id} 时出错: {e}")
#                 continue
#
#     # 计算指标
#     if all_preds:
#         rmse = np.sqrt(mean_squared_error(all_actuals, all_preds))
#         mae = mean_absolute_error(all_actuals, all_preds)
#         print(f"\nRMSE计算完成: {rmse:.4f}")
#         print(f"MAE计算完成: {mae:.4f}")
#         print(f"共处理 {len(all_preds)} 个有效预测")
#     else:
#         rmse = mae = 0
#         print("警告: 没有有效的预测评分可用于计算RMSE和MAE")
#
#     # 计算Precision@K和Recall@K
#     precision_sum = 0
#     recall_sum = 0
#     user_count = 0
#
#     # 对每个用户计算Precision@K和Recall@K
#     test_users = filtered_test['UserID'].unique()
#     print(f"\n开始计算 Precision@K 和 Recall@K，共 {len(test_users)} 个用户")
#
#     for idx, user_id in enumerate(test_users):
#         if idx % 100 == 0:
#             print(f"已处理 {idx}/{len(test_users)} 个用户")
#
#         try:
#             # 获取用户的测试评分
#             user_test_ratings = filtered_test[filtered_test['UserID'] == user_id]
#             user_high_ratings = user_test_ratings[user_test_ratings['Rating'] >= 4]['MovieID'].values
#
#             if len(user_high_ratings) == 0:
#                 # print(f"用户 {user_id} 没有高评分电影，跳过")
#                 continue
#
#             # 获取推荐
#             print(f"为用户 {user_id} 生成推荐...")
#             recommendations = get_top_n_recommendations(
#                 user_id, user_movie_ratings, user_similarity_df, movies_df, n_similar_users, top_n
#             )
#
#             if recommendations.empty:
#                 print(f"用户 {user_id} 没有推荐结果，跳过")
#                 continue
#
#             recommended_movies = recommendations['MovieID'].values
#
#             # 计算相关项目
#             relevant_items = set(user_high_ratings)
#             recommended_items = set(recommended_movies)
#
#             # 计算Precision和Recall
#             if len(recommended_items) > 0:
#                 precision = len(relevant_items.intersection(recommended_items)) / len(recommended_items)
#                 recall = len(relevant_items.intersection(recommended_items)) / len(relevant_items)
#
#                 precision_sum += precision
#                 recall_sum += recall
#                 user_count += 1
#
#                 # 打印前几个用户的详细结果
#                 if user_count <= 5:
#                     print(f"用户 {user_id} 的推荐结果:")
#                     print(f"  高评分电影: {list(relevant_items)[:5]}{'...' if len(relevant_items) > 5 else ''}")
#                     print(f"  推荐电影: {list(recommended_items)[:5]}{'...' if len(recommended_items) > 5 else ''}")
#                     print(f"  交集: {list(relevant_items.intersection(recommended_items))}")
#                     print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}")
#         except Exception as e:
#             print(f"处理用户 {user_id} 时出错: {e}")
#             continue
#
#     # 计算平均值
#     precision_at_k = precision_sum / user_count if user_count > 0 else 0
#     recall_at_k = recall_sum / user_count if user_count > 0 else 0
#     f1_score = 2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k) if (
#                                                                                                   precision_at_k + recall_at_k) > 0 else 0
#
#     print("\n" + "=" * 50)
#     print("评估完成")
#     print("=" * 50)
#     print(f"RMSE: {rmse:.4f}")
#     print(f"MAE: {mae:.4f}")
#     print(f"Precision@{top_n}: {precision_at_k:.4f}")
#     print(f"Recall@{top_n}: {recall_at_k:.4f}")
#     print(f"F1-Score@{top_n}: {f1_score:.4f}")
#     print(f"评估用户数: {user_count}")
#
#     return {
#         'RMSE': rmse,
#         'MAE': mae,
#         f'Precision@{top_n}': precision_at_k,
#         f'Recall@{top_n}': recall_at_k,
#         f'F1-Score@{top_n}': f1_score,
#         'Evaluated_Users': user_count
#     }
#
#
# # 6. 主函数
# def main():
#     """主函数"""
#     print("加载MovieLens数据集...")
#     ratings_df, users_df, movies_df = load_movielens_data()
#
#     print("预处理数据...")
#     ratings_df, user_movie_ratings = preprocess_data(ratings_df, users_df, movies_df)
#
#     print("划分训练集和测试集...")
#     train_ratings, test_ratings = train_test_split(
#         ratings_df, test_size=0.2, random_state=42
#     )
#
#     # 创建训练集的用户-电影矩阵
#     train_user_movie_ratings = train_ratings.pivot_table(
#         index='UserID', columns='MovieID', values='Rating'
#     ).fillna(0)
#
#     print("计算用户相似度...")
#     user_similarity_df = calculate_user_similarity(train_user_movie_ratings)
#
#     # 为测试用户生成推荐示例
#     test_user_id = test_ratings['UserID'].iloc[0]
#     print(f"\n为用户 {test_user_id} 生成推荐:")
#     recommendations = get_top_n_recommendations(
#         test_user_id, train_user_movie_ratings, user_similarity_df, movies_df
#     )
#     print(recommendations[['Title', 'Genres', 'Predicted_Rating']])
#
#     # 评估模型
#     print("\n评估模型性能...")
#     evaluation_results = evaluate_recommendations(
#         test_ratings, train_user_movie_ratings, user_similarity_df, movies_df
#     )
#
#     print("\n评估结果:")
#     for metric, value in evaluation_results.items():
#         print(f"{metric}: {value:.4f}")
#
#
# # 7. 运行主函数
# if __name__ == "__main__":
#     main()