import torch
import torch.nn as nn
from layers.Mamba_EncDec import Encoder, EncoderLayer
from layers.Embed import DataEmbedding_inverted

from mamba_ssm import Mamba

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = False
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.class_strategy = 0
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                        Mamba(
                            d_model=configs.d_model,  # Model dimension d_model
                            d_state=4,  # SSM state expansion factor
                            d_conv=2,  # Local convolution width
                            expand=1,  # Block expansion factor
                        ),
                        Mamba(
                            d_model=configs.d_model,  # Model dimension d_model
                            d_state=4,  # SSM state expansion factor
                            d_conv=2,  # Local convolution width
                            expand=1,  # Block expansion factor
                        ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape  # B L N

        # Embedding
        # B L N -> B L E
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # 保持原始形状

        # B L E -> B N E
        enc_out, attns = self.encoder(enc_out, attn_mask=None)  # Encoder处理

        # B L E -> B N S -> B S N
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]  # filter the covariates

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]


#假设 configs 是一个简单的命名空间或字典，包含必要的配置参数
# class Config:
#     seq_len = 96          # 输入序列长度
#     pred_len = 95           # 预测长度
#     d_model = 96           # 模型维度
#     d_state = 8            # SSM 状态扩展因子
#     embed = 'linear'       # 嵌入方式
#     freq = 'h'             # 时间频率
#     dropout = 0.1          # dropout 概率
#     d_ff = 32              # 前馈层维度
#     activation = 'relu'    # 激活函数
#     e_layers = 2           # 编码器层数
#     use_norm = True        # 是否使用归一化
#     output_attention = False
#     class_strategy = 1
#
# configs = Config()
#
# # 实例化模型
# model = Model(configs)
#
# # 创建一个随机输入 Tensor
# # 输入形状为 (批量大小, 序列长度, 特征维度)
# batch_size = 4
# input_tensor = torch.randn(batch_size, configs.d_model,configs.seq_len)  # [B, L, N]
# x_mark_enc = torch.randint(0, 1, (batch_size, configs.seq_len, 4))  # 时间标记（可选）
# x_dec = None  # 解码器输入（可选）
# x_mark_dec = None  # 解码器时间标记（可选）
#
# # 测试模型的输出
# output = model(input_tensor, x_mark_enc, x_dec, x_mark_dec)
#
# # 打印输出形状
# print("Output shape:", output.shape)  # 应输出 (batch_size, pred_len, d_model)