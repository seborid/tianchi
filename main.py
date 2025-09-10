import pandas as pd

def generate_submission():
    """
    生成基于规则的推荐结果。
    规则：选择在预测日前一天（12月18日）将商品加入购物车的用户-商品对，
          且该商品必须属于商品子集P。
    采用分块读取和优化数据类型的方式，降低内存消耗。
    """
    print("开始加载数据...")
    # 1. 加载商品子集数据，这个文件通常较小，可以一次性加载
    item_file = './tianchi_fresh_comp_train_item_online.txt'
    try:
        df_item = pd.read_csv(item_file,sep='\t',header=None,names=['item_id','item_geohash','item_category'])
        p_items = set(df_item['item_id'])
        print(f"商品子集P中共有 {len(p_items)} 个独立商品。")
    except FileNotFoundError:
        print(f"错误：请确保 '{item_file}' 文件在当前目录下。")
        return

    # 2. 分块处理用户行为数据
    user_file = './tianchi_fresh_comp_train_user_online_partA.txt'
    chunk_size = 1000000  # 每次处理100万行，可以根据你的内存大小调整
    
    # 定义读取时的数据类型，减少内存占用
    dtypes = {
        'user_id': 'int32',
        'item_id': 'int32',
        'behavior_type': 'int8',
    }
    
    # 用于存储每个分块处理后的结果
    processed_chunks = []

    print("开始分块处理用户行为数据...")
    try:
        # 使用 chunksize 进行分块读取
        # pd.read_csv 同样适用于逗号分隔的 .txt 文件
        reader = pd.read_csv(
            user_file, 
            chunksize=chunk_size,
            dtype=dtypes,
            header=None,
            sep='\t',
            names=['user_id','item_id','behavior_type','user_geohash','item_category','time']
        )

        for i, chunk in enumerate(reader):
            print(f"  正在处理分块 {i+1}...")
            
            # 筛选出预测日前一天（12-18）的行为数据
            chunk_predict_day = chunk[chunk['time'].str.startswith('2014-12-18')]
            
            # 筛选出“加购物车”（behavior_type=3）的行为
            chunk_cart = chunk_predict_day[chunk_predict_day['behavior_type'] == 3]
            
            # 只保留 user_id 和 item_id
            if not chunk_cart.empty:
                processed_chunks.append(chunk_cart[['user_id', 'item_id']])

    except FileNotFoundError:
        print(f"错误：请确保 '{user_file}' 文件在当前目录下。")
        return

    if not processed_chunks:
        print("警告：在12月18日没有找到任何'加购物车'的行为。将生成一个空的提交文件。")
        final_recommendations = pd.DataFrame(columns=['user_id', 'item_id'])
    else:
        # 合并所有处理过的分块结果
        print("合并处理结果...")
        recommendations = pd.concat(processed_chunks)
        
        # 过滤掉不在商品子集P中的商品
        recommendations_in_p = recommendations[recommendations['item_id'].isin(p_items)]
        print(f"候选推荐中，有 {len(recommendations_in_p)} 条属于商品子集P。")

        # 去重，因为一天内用户可能对同一商品多次加购物车
        final_recommendations = recommendations_in_p.drop_duplicates()
        print(f"去重后，最终生成 {len(final_recommendations)} 条推荐。")

    # 保存为指定格式的提交文件
    output_filename = 'tianchi_mobile_recommendation_predict.csv'
    final_recommendations.to_csv(
        output_filename,
        sep='\t',
        index=False,
        header=False
    )
    print(f"推荐结果已保存到文件: {output_filename}")


if __name__ == '__main__':
    generate_submission()