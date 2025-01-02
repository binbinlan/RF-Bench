import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import pandas as pd


result_dir = r'D:\new_models\Patch_Transformer\PatchTST-main\PatchTST_supervised\results\without_attribute_inverted_Informer_Runoff_ftMS_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_test_0'
# output_path = os.path.join(os.path.dirname(result_dir), 'real_prediction.csv')
# result = np.load(result_dir,allow_pickle=True)
# result = result.squeeze(0)
# print(result.shape,'\n',result)
# with open(output_path, 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     for row in result:
#         writer.writerow(row)


def t1_predict(dir,length=500):
    plt.figure()
    preds = np.load(result_dir + '/'+ 'pred.npy')[length-500:length,:,:]
    true = np.load(result_dir + '/' + 'true.npy')[length-500:length,:,:]
    preds_series = np.zeros((preds.shape[0], 1))
    true_series = np.zeros((true.shape[0], 1))
    for i in range(preds.shape[0]):
        preds_series[i] = preds[i,0,0]
        true_series[i] = true[i, 0, 0]
    plt.plot(preds_series, label='Prediction', linewidth=1)
    plt.plot(true_series, label='GroundTruth', linewidth=1)
    plt.legend()
    plt.savefig(dir + '/' + 't+1.pdf', bbox_inches='tight')

def t3_predict(dir,length=500):
    plt.figure()
    preds = np.load(result_dir + '/'+ 'pred.npy')[length-500:length,:,:]
    true = np.load(result_dir + '/' + 'true.npy')[length-500:length,:,:]
    preds_series = np.zeros((preds.shape[0],1))
    true_series = np.zeros((true.shape[0], 1))
    for i in range(preds.shape[0]):
        preds_series[i] = preds[i,5,0]
        true_series[i] = true[i, 5, 0]
    plt.plot(preds_series, label='Prediction', linewidth=1)
    plt.plot(true_series, label='GroundTruth', linewidth=1)
    plt.legend()
    plt.savefig(dir + '/' + 't+5.pdf', bbox_inches='tight')

def t5_predict(dir,length=500):
    plt.figure()
    preds = np.load(result_dir + '/'+ 'pred.npy')[length-500:length,:,:]
    true = np.load(result_dir + '/' + 'true.npy')[length-500:length,:,:]
    preds_series = np.zeros((preds.shape[0],1))
    true_series = np.zeros((true.shape[0], 1))
    for i in range(preds.shape[0]):
        preds_series[i] = preds[i,10,0]
        true_series[i] = true[i, 10, 0]
    plt.plot(preds_series, label='Prediction', linewidth=1)
    plt.plot(true_series, label='GroundTruth', linewidth=1)
    plt.legend()
    plt.savefig(dir + '/' + 't+10.pdf', bbox_inches='tight')

# t1_predict(result_dir,500)
# t3_predict(result_dir,500)
# t5_predict(result_dir,500)

# def export_result(dir,start=0,end=1291,time=1):
#     preds = np.load(result_dir + '/' + 'pred.npy')[start:end, :, :]
#     true = np.load(result_dir + '/' + 'true.npy')[start:end, :, :]
#     preds_series = np.zeros((preds.shape[0], 1))
#     true_series = np.zeros((true.shape[0], 1))
#     for i in range(preds.shape[0]):
#         preds_series[i] = preds[i, time-1, 0]
#         true_series[i] = true[i, time-1, 0]
#     np.savetxt(dir + '/' +'pred.txt',preds_series)
#     np.savetxt(dir + '/' + 'true.txt', true_series)
#
#
# export_result(result_dir,start=0,end=2000,time=1)



def export_all_results(dir,time=1):

    pred = np.load(dir + '/' + 'pred.npy')
    true = np.load(dir+ '/' + 'true.npy')

    excel_file = r'D:\new_models\Patch_Transformer\PatchTST-main\PatchTST_supervised\data\basin_information.xlsx'  # 替换为实际文件名
    df = pd.read_excel(excel_file)

    t_folder = os.path.join(dir,str(time))

    if not os.path.exists(t_folder):
        os.makedirs(t_folder)

    num_samples = true.shape[0]
    sample_size = 2000

    # 计算总样本数
    total_samples = num_samples // sample_size

    for i in range(total_samples):
        # 计算当前样本的起始索引
        start_index = i * sample_size
        end_index = start_index + sample_size

        # 提取当前样本
        true_sample = true[start_index:end_index,time,:] *  float(df.iloc[i, 0]) / (1000) # (2000, t, 1)
        pred_sample = pred[start_index:end_index, time, :] * float(df.iloc[i, 0]) / (1000)

        pred_sample[pred_sample < 0] = 0
        true_sample[true_sample < 0] = 0

        X = np.arange(true_sample.shape[0])

        plt.figure(figsize=(10, 5))
        plt.plot(X, true_sample, label=f'observation at lead time {time} hour',linestyle='--', color='blue')
        plt.plot(X, pred_sample, label=f'Prediction at lead time {time} hour',  color='orange')

        plt.xlabel('time')
        plt.ylabel('streamflow value (m^3/h)')
        plt.title(df.iloc[i, 0])
        plt.legend(loc='upper left')
        plt.savefig(os.path.join(t_folder, f'{df.iloc[i, 0]}.png'))
        plt.close()  # 关闭当前图形，释放内存



export_all_results(result_dir,time=6)