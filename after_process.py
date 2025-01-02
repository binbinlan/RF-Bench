
import torch
import os
from exp.exp_main import Exp_Main

result_dir = r'D:\new_models\Patch_Transformer\PatchTST-main\PatchTST_supervised\checkpoints\without_attribute_inverted_24h_Informer_Runoff_ftMS_sl96_ll48_pl24_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_test_0'

def delibrate_model_name(result_dir):


    path = os.path.join(result_dir,'checkpoint.pth')

    model_dic = torch.load(path)

    model_dic =  {k.replace("module.", ""): v for k, v in model_dic.items()}

    try:
        torch.save(model_dic, path)
        print(f"模型成功保存到: {path}")
    except Exception as e:
        raise RuntimeError(f"保存模型失败: {e}")

    print("处理完成。")

delibrate_model_name(result_dir)