import numpy as np
#
# # 读取.npy文件
ob1 = np.load(r'D:\new_models\Patch_Transformer\PatchTST-main\PatchTST_supervised\results\without_attribute_DLinear_Runoff_ftMS_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_test_0\true.npy')
# ob2 = np.load(r'D:\new_models\Patch_Transformer\PatchTST-main\PatchTST_supervised\results\Runoff_test100_PatchTST_Runoff_ftMS_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_test_0\true.npy')
# ob3 = np.load(r'D:\new_models\Patch_Transformer\PatchTST-main\PatchTST_supervised\results\Runoff_test100_With_attri_PatchTST_Runoff_ftMS_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_test_0\true.npy')
# # 打印数组
#
print(ob1.shape,)
#

# import numpy as np
# from sklearn.preprocessing import StandardScaler
#
# # 创建一个示例数据集
# data = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
# data2 = np.array([[1.2], [3.0], [1.0], [2.7], [-2.0]])
# # 初始化 StandardScaler
# scaler = StandardScaler()
#
# # 在训练数据上拟合并转换
# scaled_data = scaler.fit_transform(data)
# scaled_data2 = scaler.fit_transform(data)
#
# # 打印归一化后的值
# print("归一化后的值:")
# print(scaled_data)
#
# # 进行反归一化
# original_data = scaler.inverse_transform(scaled_data)
# original_data2 = scaler.inverse_transform(scaled_data2)
#
# # 打印反归一化后的值
# print("反归一化后的值:")
# print(original_data)
# print("反归一化后的值2:")
# print(original_data2)
#
# import pandas as pd
#
# # 读取 Excel 文件
# camles_topo = pd.read_excel(r'D:\短期工作\2024_09\model\camels_topo.xlsx')
# basin_information = pd.read_excel(r'D:\短期工作\2024_09\model\basin_information.xlsx')
#
# # 合并数据
# merged_df = basin_information.merge(camles_topo[['gauge_id', 'area_gages2']],
#                                      left_on='filename', right_on='gauge_id',
#                                      how='left')
#
# # 将 area_gauge 列添加到 basin_information DataFrame
# basin_information['area_gages2'] = merged_df['area_gages2']
#
# # 保存到 basin_information.xlsx
# basin_information.to_excel('basin_information.xlsx', index=False)
#
# print("更新完成，area_gauge 列已添加到 basin_information.xlsx。")