import os
import pandas as pd
from tqdm import tqdm

def merge_data(folder1, folder2, output_folder):
    # 遍历文件夹1内的所有文件
    station_ids = []
    for filename in tqdm(os.listdir(folder1)):
        if filename.endswith('.csv'):
            # 提取水文站ID
            station_id = filename.split('-')[0]
            station_ids.append(station_id)

            # 读取文件夹1内的文件
            file1 = os.path.join(folder1, filename)
            df1 = pd.read_csv(file1, usecols=['date', 'QObs(mm/h)'])

            # 读取文件夹2内对应的文件
            file2 = os.path.join(folder2, f'{station_id}_hourly_nldas.csv')

            if not os.path.exists(file2):
                continue

            df2 = pd.read_csv(file2)

            # 合并两个数据框
            merged_df = pd.merge(df2, df1, on='date')
            merged_df = merged_df.dropna(subset=['QObs(mm/h)']).head(10000)

            # 保存到新的文件夹内，以水文站ID命名
            output_file = os.path.join(output_folder, f'{station_id}.csv')
            merged_df.to_csv(output_file, index=False)

    # 保存 station_id 到文本文件
    station_ids_file = os.path.join(output_folder, 'station_ids.txt')
    with open(station_ids_file, 'w') as f:
        f.write('\n'.join(station_ids))

folder1 = r'D:\hourly_lstm\hourly_data\usgs_streamflow'
folder2 = r'D:\hourly_lstm\hourly_data\nldas_hourly'
output_folder = r'D:\new_models\Patch_Transformer\PatchTST-main\PatchTST_supervised\data\Runoff_hour'

merge_data(folder1, folder2, output_folder)