# 把一个文件夹里面的所有的.pkl文件转换成parquet文件
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
def convert_pkl_to_parquet(folder_path):
    pkl_files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]
    for file_name in tqdm(pkl_files):
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_pickle(file_path, compression='infer')
        parquet_file_name = file_name.replace('.pkl', '.parquet')
        parquet_file_path = os.path.join(folder_path, parquet_file_name)
        df.to_parquet(parquet_file_path, index=False)
        print(f"Converted {file_name} to {parquet_file_name}")

if __name__ == "__main__":
    folder_path = 'output/chn'
    convert_pkl_to_parquet(folder_path)
    folder_path = 'output/eng'
    convert_pkl_to_parquet(folder_path)