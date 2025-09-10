import pandas as pd

def data_processing():
    item_file = './tianchi_fresh_comp_train_item_online.txt'
    try:
        df_item = pd.read_csv(item_file,sep='\t',header=None,names=['item_id','item_geohash','item_category'])
    except FileNotFoundError:
        print(f"错误：请确保 '{item_file}' 文件在当前目录下。")
    
    return   
