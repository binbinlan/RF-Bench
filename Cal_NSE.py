import os.path
from typing import List
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from collections import defaultdict
import matplotlib.pyplot as plt

result_dir = r'D:\new_models\Patch_Transformer\PatchTST-main\PatchTST_supervised\results\Runoff_test100_Transformer_Runoff_ftMS_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_test_0'
location_dir = r"D:\CAMELS\data\Caravan\attributes\camels\attributes_other_camels.csv"

def add_zero_prefix(filename):
    filename = str(filename)
    if len(filename) == 7:
        return '0' + filename
    else:
        return filename

def get_basin_list() -> List:
    """Read list of basins from text file.

    Returns
    -------
    List
        List containing the 8-digit basin code of all basins
    """
    basin_file = Path(r"D:\new_models\Patch_Transformer\PatchTST-main\PatchTST_supervised\data\Runoff\station_ids.txt")
    with basin_file.open('r') as fp:
        basins = fp.readlines()
    basins = [basin.strip() for basin in basins]

    return basins

def cal_NSE(lead_time = 12):
    NSE_list = []
    file_list = []
    out_dir = Path(result_dir)
    pred_list = []
    pred = np.load(result_dir + '//' + 'pred.npy')
    obs = np.load(result_dir + '//' + 'true.npy')


    pred_values = pred[:,lead_time-1,0]
    obs_values = obs[:,lead_time-1,0]
    basin_lists = get_basin_list()

    for i in range(516):

        pred_value = pred_values[i*2000:(i+1)*2000]
        obs_value = obs_values[i*2000:(i+1)*2000]

        numerator = np.sum((obs_value - pred_value) ** 2)
        denominator = np.sum((obs_value - np.mean(obs_value)) ** 2)

        if denominator != 0:
            nse = 1 - (numerator / denominator)
            NSE_list.append(nse)
            file_list.append(basin_lists[i])

    df2 =  pd.DataFrame({'filename': file_list, 'NSE': NSE_list})
    df2['filename'] = df2['filename'].astype(str)
    save_name = "{}{}".format(str(lead_time), 'NSE.csv')
    df2.to_csv(os.path.join(result_dir, save_name), index=False, quoting=3)

def cal_TPE(lead_time = 12):
    TPE_list = []
    file_list = []
    out_dir = Path(result_dir)
    pred_list = []
    pred = np.load(result_dir + '//' + 'pred.npy')
    obs = np.load(result_dir + '//' + 'true.npy')


    pred_values = pred[:,:,0]
    obs_values = obs[:,:,0]
    basin_lists = get_basin_list()

    for i in range(516):

        pred_value = pred_values[i*2000:(i+1)*2000]
        obs_value = obs_values[i*2000:(i+1)*2000]

        observed = obs_value[:, lead_time - 1]

        # 找到最大2%的观测流量值的索引
        top_2_percent = int(len(observed) * 0.02)
        top_2_percent_indices = np.argsort(observed)[-top_2_percent:]  # 获取最大2%的索引

        # 计算对应的预测值
        corresponding_pred_values = pred_value[top_2_percent_indices, lead_time - 1]

        # 计算误差
        errors = np.abs(corresponding_pred_values - observed[top_2_percent_indices])
        top_2_percent_error = np.sum(errors) / np.sum(observed[top_2_percent_indices])

        TPE_list.append(top_2_percent_error)
        file_list.append(basin_lists[i])

    df2 =  pd.DataFrame({'filename': file_list, 'TPE': TPE_list})
    df2['filename'] = df2['filename'].astype(str)
    save_name = "{}{}".format(str(lead_time), 'TPE.csv')
    df2.to_csv(os.path.join(result_dir, save_name), index=False, quoting=3)

def plot_NSE_distribution(ed_dirs,region,lead_time):
    #columns_name = ['gauge_id','gauge_lat','gauge_lon']
    df_nse = pd.read_csv(Path(result_dir + '//'+ 'NSE.csv'),dtype={'filename:':str})
    df_attributes= pd.read_csv(Path(location_dir))

    data1 = gpd.read_file(r'D:\terrain\nature_earth\10m_cultural\ne_10m_admin_0_countries.shp')
    data2 = gpd.read_file(r'D:\terrain\nature_earth\10m_cultural\ne_10m_admin_1_states_provinces_lines.shp')
    data3 = gpd.read_file(r'D:\terrain\nature_earth\10m_cultural\ne_10m_admin_0_boundary_lines_land.shp')

    mask = r'D:\terrain\country_shape\World_countries.shp'
    mask = gpd.read_file(mask)
    country_name = 'UNITED STATES'
    mask = mask[mask['NAME'].isin(
        [country_name])]
    data1 = gpd.overlay(data1, mask)
    data2 = gpd.overlay(data2, mask)
    data3 = gpd.overlay(data3, mask)
    fig, ax = plt.subplots()
    data1.plot(ax=ax, color='lightsteelblue', linewidth=0.5)
    data2.plot(ax=ax, color='gray', linewidth=0.5)
    data3.plot(ax=ax, color='black', linewidth=2)


    data = defaultdict(list)
    df_attributes['gauge_id'] = df_attributes['gauge_id'].str.strip('camels_')
    df_nse['filename'] = df_nse['filename'].apply(add_zero_prefix)
    merged_df = pd.merge(df_attributes, df_nse, left_on='gauge_id', right_on='filename')

    print(merged_df)
    print(df_attributes['gauge_id'])
    print(df_nse['filename'])


    scatter = ax.scatter(x=merged_df['gauge_lon'], y=merged_df['gauge_lat'], c=merged_df['NSE'], cmap='viridis', s=50,
                         edgecolor='black', linewidth=0.5, vmin=-1, vmax=1)

    colorbar = plt.colorbar(scatter)
    plt.show()

    #print(gauges)

for i in [1,3,6,9,12,24]:
    cal_TPE(lead_time = i)
#plot_NSE_distribution(1,2,3)



