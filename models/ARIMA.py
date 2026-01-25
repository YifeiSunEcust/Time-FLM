import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

from data_provider.data_factory import data_provider
from models import Autoformer, DLinear, TimeLLM, TimesNet, PatchTST, Informer, iTransformer, \
    TimeXer, TiDE, MtsLLM, woLLM, TLinear, ARIMA
from tqdm import tqdm
import time as t
import torch
from torch import nn, optim
import argparse
import os
import random
import numpy as np
import pandas as pd
from utils.tools import validate
from utils.metrics import metric
os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description='Time-LLM')
# basic config
parser.add_argument('--model', type=str, default='ARIMA',
                    help='model name, options: [TimeLLM, Autoformer, DLinear, TLinear, TimesNet, PatchTST, Informer, iTransformer'
                         'TimeXer, TiDE, MtsLLM, woLLM]')
parser.add_argument('--task_name', type=str, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')

# data loader
parser.add_argument('--data', type=str, default='EFP_long', help='dataset type')
parser.add_argument('--root_path', type=str, default='F:\Time-LLM\Time-LLM-main\dataset\EFP', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='EFP_long.csv', help='data file')
parser.add_argument('--target', type=str, default='hx', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, '
                         'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                         'you can also use more detailed freq like 15min or 3h')
parser.add_argument('--features', type=str, default='M',help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S: univariate predict univariate, MS:multivariate predict univariate')
# model define
parser.add_argument('--enc_in', type=int, default=24, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
parser.add_argument('--c_out', type=int, default=1, help='output size')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_model', type=int, default=32, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
parser.add_argument('--d_ff', type=int, default=128, help='dimension of fcn')
parser.add_argument('--top_k', type=int, default=5, help='TimesNet top_k')
parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
parser.add_argument('--channel_independence', type=int, default=1, help='0: channel dependence 1: channel independence for FreTS model')
parser.add_argument('--decomp_method', type=str, default='moving_avg', help='method of series decompsition, only support moving_avg or dft_decomp')
parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
parser.add_argument('--down_sampling_method', type=str, default=None, help='down sampling method, only support avg, max, conv')
parser.add_argument('--patch_len', type=int, default=8, help='patch length')
parser.add_argument('--stride', type=int, default=4, help='stride')
parser.add_argument('--prompt_domain', type=int, default=0, help='')
parser.add_argument('--llm_model', type=str, default='GPT2', help='LLM model') # LLAMA, GPT2, BERT
parser.add_argument('--llm_dim', type=int, default='768', help='LLM model dimension')# LLama7b:4096; GPT2-small:768; BERT-base:768
parser.add_argument('--llm_layers', type=int, default=6)

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=24, help='start token length')
parser.add_argument('--pred_len', type=int, default=12, help='prediction sequence length')

# optimization
parser.add_argument('--batch_size', type=int, default=4, help='batch size of train input data')
parser.add_argument('--train_epochs', type=int, default=32, help='train epochs')
parser.add_argument('--num_experiments', type=int, default=2, help='Number of training sessions')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
args = parser.parse_args()

class Model(nn.Module):
    def __init__(self, configs, p=1, d=1, q=1):
        super(Model, self).__init__()
        self.p = p
        self.d = d
        self.q = q
        self.pre_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in

    def forward(self, x):
        """
        使用ARIMA模型对每个序列的每个变量进行预测
        :param x: 输入张量，形状为 [batch, input_timestamps, variables]
        :param steps: 预测的时间步长
        :return: 预测张量，形状为 [batch, steps, variables]
        """
        B, _, _ = x.shape
        forecast_results = []

        for b in range(B):
            batch_forecast = []
            for v in range(self.enc_in):
                series = x[b, :, v].cpu().numpy() if isinstance(x[b, :, v], torch.Tensor) else x[b, :, v]

                # 拟合ARIMA模型
                try:
                    arimamodel = ARIMA(series, order=(self.p, self.d, self.q))
                    fitted_model = arimamodel.fit()
                    forecast = fitted_model.forecast(steps=self.pre_len)
                except Exception as e:
                    # 如果ARIMA模型拟合失败，可以用简单的方法填充
                    print(f"ARIMA fitting failed for batch {b}, variable {v}: {e}")
                    forecast = np.zeros(self.pre_len)

                batch_forecast.append(forecast)

            # 转换为形状为 [steps, variables] 的预测
            batch_forecast = np.array(batch_forecast).T
            forecast_results.append(batch_forecast)

        # 转换为张量，形状为 [batch, steps, variables]
        forecast_results = torch.tensor(forecast_results, dtype=torch.float32)
        return forecast_results


# 示例用法
if __name__ == "__main__":
    # 模拟数据：batch为2，input_timestamps为10，variables为3
    train_data, train_loader = data_provider(args, 'train')
    vali_data, vali_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')
    batch, input_timestamps, variables = 2, 10, 3
    x = torch.randn(batch, input_timestamps, variables)  # 随机生成一些数据

    # 初始化ARIMA模型，参数设定为(p=1, d=1, q=1)
    arima_model = Model(p=1, d=1, q=1)

    # 使用模型进行多步预测，假设预测5步
    steps = args.pred_len
    forecast = arima_model(x, steps=steps)

    print("Forecast results:")
    print(forecast)
