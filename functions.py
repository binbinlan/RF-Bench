import numpy as np
import os

dir_path = r'D:\new_models\Patch_Transformer\PatchTST-main\PatchTST_supervised\results\without_attribute_LSTM_Runoff_ftMS_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_test_0'
lead_days = 23

pred = np.load(os.path.join(dir_path,'pred.npy'))
obs = np.load(os.path.join(dir_path,'true.npy'))

def cal_NSE(lead_days):
    obs_mean = np.mean(obs[:,lead_days,:])
    nse = 1 - np.sum((pred[:,lead_days,:] - obs[:,lead_days,:])**2) / np.sum((obs[:,lead_days,:] - obs_mean)**2)
    return nse


def cal_KGE(lead_days):
    # coefficient
    r = np.corrcoef(obs[:,lead_days,:].squeeze(),pred[:,lead_days,:].squeeze())[0, 1]
    sigma_p = np.std(pred[:,lead_days,:])
    sigma_o = np.std(obs[:,lead_days,:])
    mu_p = np.mean(pred[:,lead_days,:])
    mu_o = np.mean(obs[:,lead_days,:])
    kge = 1 - np.sqrt((r - 1) ** 2 + (sigma_p / sigma_o - 1) ** 2 + (mu_p / mu_o - 1) ** 2)
    return kge

def cal_TPE(lead_days):
    # 2% peak flow error
    errors = np.abs(obs[:,lead_days,:] - pred[:,lead_days,:])
    errors = np.sort(errors)[::-1]

    observed = np.array(obs[:,lead_days,:])
    observed = observed[np.argsort(errors)[:int(len(errors) * 0.02)]]

    top_2_percent = int(len(errors) * 0.02)
    top_2_percent_error = np.sum(errors[:top_2_percent]) / np.sum(observed)
    return top_2_percent_error

def cal_R2(lead_days):
    obs_mean = np.mean(obs[:,lead_days,:])
    pred_mean = np.mean(pred[:,lead_days,:])
    numerator = np.sum((obs[:,lead_days,:] - obs_mean) * (pred[:,lead_days,:] - pred_mean)) ** 2
    denominator = np.sum((obs[:,lead_days,:] - obs_mean)**2) * np.sum((pred[:,lead_days,:] - pred_mean)**2)
    R2 = numerator / denominator
    return R2

def cal_MAE(lead_days):
    mae = np.abs(obs[:,lead_days,:] - pred[:,lead_days,:])
    return np.mean(mae)

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
        print(nse)


    return sum(NSE_list)/len(NSE_list)


nse = cal_NSE(lead_days)
kge = cal_KGE(lead_days)
TPE = cal_TPE(lead_days)
R2 = cal_R2(lead_days)
mae = cal_MAE(lead_days)
#every_nse = cal_every_NSE(lead_days)

print("nse:", nse,"kge:",kge,'TPE:',TPE,'R2:',R2,"MAE:",mae)
