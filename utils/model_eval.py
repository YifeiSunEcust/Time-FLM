import os
import random
import numpy as np
import pandas as pd

from data_provider.FLMdata_factory import data_provider
from models import FLM, FLM_Debug, FLM_Trans
from tqdm import tqdm
import time as t
import torch
from torch import nn, optim
import argparse
from train import train_model
from validate import evaluate_model
os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description='Fermentation Large Model')
# basic config
parser.add_argument('--model', type=str, default='FLM',
                    help='model name, options: [FLM]')

# data loader
parser.add_argument('--data', type=str, default='FLM', help='dataset type')
parser.add_argument('--root_path', type=str, default='F:\FLM\datasets\FLM_datasets', help='root path of the data file')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, '
                         'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                         'you can also use more detailed freq like 15min or 3h')
parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S: univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--use_dynamic_batch', type=bool, default=True,
                    help='Use dynamic batch sampling based on sequence length')
parser.add_argument('--total_range', type=tuple, default=[8, 64],
                    help='Sequence length range for input [8, 128]=[8, 16], [16, 32], [32, 64], [64, 128]')
parser.add_argument('--num_workers', type=int, default=8, help='num of encoder layers')
# model define
parser.add_argument('--e_layers', type=int, default=6, help='num of encoder layers')
parser.add_argument('--num_patch', type=int, default=40, help='num of encoder layers')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--d_ff', type=int, default=512, help='dimension of fcn')
parser.add_argument('--num_experts', type=int, default=4, help='num of experts')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')

parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')

# optimization
parser.add_argument('--batch_size', type=int, default=256, help='batch size of train input data')
parser.add_argument('--train_epochs', type=int, default=5, help='train epochs')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='optimizer learning rate')
args = parser.parse_args()


def generate_predictions(model, src, max_len=50):
    model.eval()
    tgt = torch.zeros(src.size(0), 1, 1).to(src.device)  # 初始化tgt (通常是开始标记)

    predictions = []
    with torch.no_grad():
        for _ in range(max_len):
            with torch.amp.autocast('cuda'):
                output = model(src, tgt)  # 输入src和当前tgt（注意：tgt是逐步生成的）
                next_token = output[:, -1, :]  # 获取最后一个时间步的输出（下一步的预测）
                predictions.append(next_token.unsqueeze(1))

                # 将预测结果作为新的tgt输入
                tgt = torch.cat([tgt, next_token.unsqueeze(1)], dim=1)

    return torch.cat(predictions, dim=1)


if __name__ == '__main__':
    # 加载模型
    if args.model == 'FLM':
        model = FLM_Trans.Model(args).float().to(device)

    checkpoint = torch.load('F:\FLM\FLM_form1_epoch2.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # 5. 获取保存的epoch和损失（如果需要恢复训练过程）
    epoch = checkpoint['epoch']
    val_loss = checkpoint['loss']
    test_loader = data_provider(args, 'val')

    for i, (src, tgt, src_mark, tgt_mark, src_mask, tgt_mask) in tqdm(enumerate(test_loader)):
        src, tgt, src_mask, tgt_mask = src.to(device), tgt.to(device), src_mask.to(device), tgt_mask.to(device)
        pre_out = generate_predictions(model, src, len(tgt[1]))
        print(pre_out)
        print(tgt)
        break
