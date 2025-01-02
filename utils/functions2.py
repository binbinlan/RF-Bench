import numpy as np
import os

dir_path = r'D:\new_models\Patch_Transformer\PatchTST-main\PatchTST_supervised\results\without_attribute_24_LSTM_Runoff_ftMS_sl96_ll48_pl24_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_test_0'
lead_days_list = [1,3,6,9,12,24]

pred = np.load(os.path.join(dir_path,'pred.npy'))
obs = np.load(os.path.join(dir_path,'true.npy'))

def cal_every_NSE(lead_days):
    NSE_list = []
    pred_values = pred[:,lead_days,0]
    obs_values = obs[:,lead_days,0]
    for i in range(516):
        pred_value = pred_values[i*2000:(i+1)*2000]
        obs_value = obs_values[i*2000:(i+1)*2000]
        numerator = np.sum((obs_value - pred_value) ** 2)
        denominator = np.sum((obs_value - np.mean(obs_value)) ** 2)
        if denominator != 0:
            nse = 1 - (numerator / denominator)
            if nse < -1:
                nse = -1
        NSE_list.append(nse)
    return sum(NSE_list)/len(NSE_list)

nse_list = []
for lead_days in lead_days_list:
    nse = cal_every_NSE(lead_days-1)
    nse_list.append(nse)

print('nse:',nse_list)