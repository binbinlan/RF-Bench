import pickle
import csv
import os
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List
from pathlib import Path
from collections import defaultdict
#from bokeh.sampledata import us_states
#us_states = us_states.data.copy()
plt.rcParams['font.family'] = 'Times New Roman'

def add_zero_prefix(filename):
    filename = str(filename)
    if len(filename) == 7:
        return '0' + filename
    else:
        return filename


result_dir = r'D:\new_models\Patch_Transformer\PatchTST-main\PatchTST_supervised\papercodes'
location_dir = r"D:\CAMELS\data\Caravan\attributes\camels\attributes_other_camels.csv"


df_nse = pd.read_csv(Path(result_dir + '//' + 'plot_all_nse_12h.csv'), dtype={'filename:': str})
df_attributes = pd.read_csv(Path(location_dir))

data1 = gpd.read_file(r'D:\terrain\nature_earth\10m_cultural\ne_10m_admin_0_countries.shp')
data2 = gpd.read_file(r'D:\terrain\nature_earth\10m_cultural\ne_10m_admin_1_states_provinces_lines.shp')
#data3 = gpd.read_file(r'D:\terrain\nature_earth\10m_cultural\ne_10m_admin_0_boundary_lines_land.shp')

mask = r'D:\terrain\country_shape\World_countries.shp'
mask = gpd.read_file(mask)
country_name = 'UNITED STATES'
mask = mask[mask['NAME'].isin(
    [country_name])]
data1 = gpd.overlay(data1, mask)
data2 = gpd.overlay(data2, mask)
#data3 = gpd.overlay(data3, mask)
fig, ax = plt.subplots()
data1.plot(ax=ax, color='lightsteelblue', linewidth=0.5)
data2.plot(ax=ax, color='gray', linewidth=0.5)
#data3.plot(ax=ax, color='black', linewidth=2)

df_attributes['gauge_id'] = df_attributes['gauge_id'].str.strip('camels_')
df_nse['filename'] = df_nse['filename'].apply(add_zero_prefix)
merged_df = pd.merge(df_attributes, df_nse, left_on='gauge_id', right_on='filename')
print(merged_df)


model_names = ['Autoformer', 'Informer', 'Dlinear', 'LSTM', 'Patch_TST', 'RRformer','Mamba']
markers = ['o', 's', '^', 'D', 'P', 'X', '*']  # 定义不同的散点形状

nse_values = merged_df.iloc[:, 7:]  # 假设第7列的索引为6
merged_df['max_nse'] = nse_values.max(axis=1)
merged_df['best_model'] = nse_values.idxmax(axis=1)
merged_df['max_nse'] = merged_df['max_nse'].clip(lower=-1)


for model, marker in zip(model_names, markers):
        model_df = merged_df[merged_df['best_model'] == model]
        scatter = ax.scatter(
            x=model_df['gauge_lon'],
            y=model_df['gauge_lat'],
            s=75,  # 固定散点大小
            c=model_df['max_nse'],  # 根据最大 NSE 值设置颜色
            cmap='viridis',
            edgecolor='black',
            linewidth=0.5,
            vmin=-1,  # 设置颜色条的最小值
            vmax=1,
            marker=marker,
            label=model
        )
colorbar = plt.colorbar(scatter,orientation='horizontal')
colorbar.set_label('Max NSE Value')

# 自定义图例
handles = [plt.Line2D([0], [0], marker=marker, color='none', markerfacecolor='gray', markersize=10) for marker in markers]
#legend = ax.legend(handles, model_names, title='Models', loc='lower right',ncol=len(model_names), frameon=True)

plt.title(f'NSE value distribution of the best model')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

best_model_counts = merged_df['best_model'].value_counts()
best_model_counts = best_model_counts.reindex(model_names, fill_value=0)  # 确保所有模型都有显示
#colors = ['tab:purple','lightcoral','gold', 'yellowgreen', 'lightskyblue','tab:red' ]
colors = [(219,49,36),(252,140,90),(255,223,146), (230,241,243), (144,190,224),(75,116,178),(38,70,83)]
colors = [(r/255, g/255, b/255) for r, g, b in colors]

print("各模型所占份额:")
for model, count in best_model_counts.items():
    percentage = (count / len(merged_df)) * 100
    print(f"{model}: {percentage:.2f}%")

max_index = best_model_counts.idxmax()
explode = [0.1 if model == max_index else 0 for model in model_names]  # 仅分离最大部分

plt.figure(figsize=(8, 6))
wedges, texts, autotexts = plt.pie(
    best_model_counts,  autopct=lambda p: f'{p:.1f}%' if p > 5 else '',
    startangle=140, colors=colors, explode=explode, shadow=True,
    wedgeprops=dict(edgecolor='white'), pctdistance=0.85  # 调整百分比文本的位置
)


# plt.figure(figsize=(8, 6))
# wedges, texts, autotexts = plt.pie(
#     best_model_counts, labels=model_names, autopct=lambda p: f'{p:.1f}%' if p > 5 else '',
#     startangle=140, colors=colors, explode=explode, shadow=True,
#     wedgeprops=dict(edgecolor='white'), pctdistance=0.85  # 调整百分比文本的位置
# )

# 设置阴影效果
for w in wedges:
    w.set_linewidth(1)  # 设置边缘线宽
    w.set_edgecolor('white')  # 设置边缘颜色

plt.setp(autotexts, size=35, weight="bold", color="white")  # 设置百分比文本样式
plt.title('Percentage of Best NSE Models')
plt.axis('equal')  # 使饼状图为圆形
plt.show()