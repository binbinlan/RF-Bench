from data_provider.data_loader import Dataset_Pred
from torch.utils.data import DataLoader
from data_provider.data_loader import Dataset_ETT_hour,Dataset_Runoff
from torch.utils.data import DataLoader
import argparse
import os
import torch
import matplotlib.pyplot as plt
import random
import numpy as np
from utils.tools import dotdict
from exp.exp_main import Exp_Main
import torch
import matplotlib
import seaborn as sns
import torch.nn.functional as F

"""
for Inverse_Informer
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

    # random seed
    parser.add_argument('--random_seed', type=int, default=2021, help='random seed')

    # basic config
    parser.add_argument('--is_training', type=int,default=0, help='status')
    parser.add_argument('--model_id', type=str, default='without_attribute_inverted_24h', help='model id')
    parser.add_argument('--model', type=str,  default='Informer',
                        help='model name, options: [Autoformer, Informer, Transformer,PatchTST]')
    parser.add_argument('--all_model', type=bool,  default=False,
                        help='run all models in a exp, options: [Autoformer, Informer, Transformer, PatchTST]')

    # data loader
    parser.add_argument('--data', type=str,  default='Runoff', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/Runoff_hour/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='./data/Runoff_hour/', help='data file')
    parser.add_argument('--features', type=str, default='MS',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='QObs(mm/h)', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument("--with_attributes",action='store_true',help='use static attributes')
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')


    # DLinear
    #parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')

    # PatchTST
    parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
    parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
    parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

    # Formers
    parser.add_argument('--embed_type', type=int, default=1, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
    parser.add_argument('--enc_in', type=int, default=12, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
    parser.add_argument('--dec_in', type=int, default=12, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=12, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention',  action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', default=True, help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=100, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

    args = parser.parse_args()
    print('Args in experiment:')
    print(args)

    # random seed
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)


    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]


Data = Dataset_Runoff
timeenc = 0 if args.embed!='timeF' else 1
flag = 'test'; shuffle_flag = True; drop_last = True; batch_size = 1

data_set = Data(
    root_path=args.root_path,
    data_path=args.data_path,
    flag=flag,
    size=[args.seq_len, args.label_len, args.pred_len],
    features=args.features,
    timeenc=timeenc,
    freq=args.freq,
    with_attributes=args.with_attributes
)
data_loader = DataLoader(
    data_set,
    batch_size=batch_size,
    shuffle=shuffle_flag,
    num_workers=args.num_workers,
    drop_last=drop_last)


args.output_attention = True

Exp = Exp_Main
exp = Exp(args)

model = exp.model
print(model)


setting = 'without_attribute_inverted_24h_Informer_Runoff_ftMS_sl96_ll48_pl24_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_test_0'

path = os.path.join(args.checkpoints,setting,'checkpoint.pth')

model_dic = torch.load(path)

model.load_state_dict(model_dic)
model.eval()

print(model)


layer = 0
args.attn = 'prob'

idx = 50

scores = np.zeros((8,16))

distil = 'Distil' if args.distil else 'NoDistil'
def visual_attention_in_col():
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
        if i <= idx:
            batch_x = batch_x.float().to(exp.device)
            batch_y = batch_y.float()

            batch_x_mark = batch_x_mark.float().to(exp.device)
            batch_y_mark = batch_y_mark.float().to(exp.device)

            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(exp.device)

            outputs, attn = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            for h in range(0,8):
                plt.figure(figsize=[10,8])
                plt.title('Informer, {}, attn:{} layer:{} head:{}'.format(distil, args.attn, layer, h))
                A = attn[layer][0,h].detach().cpu().numpy()
                scores[h] = scores[h] + np.sum(A, axis=0)
                ax = sns.heatmap(A, vmin=0, vmax=A.max()+0.01)
                plt.gca().invert_yaxis()
                plt.savefig('attentions/Informer_24h/sample_{}_head_{}.png'.format(i, h))  # 保存图像到当前目录
                plt.close()
        else:
            break
    print(scores.shape)
    np.savetxt('attentions/Informer_24h/data.txt',scores)

def visual_attention_in_row():
    # 初始化分数列表，8个头，每个头的分数初始化为0
    scores = np.zeros((8, 16))

    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
        if i <= idx:
            # 数据处理
            batch_x = batch_x.float().to(exp.device)
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float().to(exp.device)
            batch_y_mark = batch_y_mark.float().to(exp.device)

            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(exp.device)

            # 模型推理
            outputs, attn = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            for h in range(8):  # 假设有8个头
                A = attn[layer][0, h].detach().cpu().numpy()

                # 计算每一行的和
                row_sums = np.sum(A, axis=1)

                # 找到最大行和的索引
                max_row_index = np.argmax(row_sums)
                print('sample{}的头{}最大索引是{}'.format(i,h,max_row_index))
                # 为最大行索引对应的头加1分
                scores[h][max_row_index] += 1  #

    # 输出每个头的最终分数
    for h in range(8):
        print(f"Scores for Head {h + 1}: {scores[h]}")

visual_attention_in_row()