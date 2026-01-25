import pandas as pd

from data_provider.FLMdata_factory import data_provider
from models import TimeFLM
from tqdm import tqdm
import time as t
import torch
from torch import nn, optim
import argparse
import os
import random
import numpy as np
import yaml

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
parser.add_argument('--e_layers', type=int, default=4, help='num of encoder layers')
parser.add_argument('--max_len', type=int, default=100, help='maximum input sequence length')
parser.add_argument('--num_patch', type=int, default=40, help='num of patchs')
parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--d_ff', type=int, default=128, help='dimension of fcn')
parser.add_argument('--expert_nums', type=int, default=6, help='num of experts')
parser.add_argument('--top_k', type=int, default=2, help='dimension of expert')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')

parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')

# optimization
parser.add_argument('--batch_size', type=int, default=256, help='batch size of train input data')
parser.add_argument('--train_epochs', type=int, default=32, help='train epochs')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
args = parser.parse_args()

# Load configuration from YAML file
config_path = 'config.yaml'
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    # Update args with config values
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)
        else:
            print(f"Warning: {key} not found in args, skipping")
else:
    print(f"Warning: {config_path} not found, using default args")


if __name__ == "__main__":

    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = data_provider(args, 'train')
    vali_loader = data_provider(args, 'val')

    if args.model == 'FLM':
        model = TimeFLM.Model(args).float().to(device)

    criterion = nn.MSELoss()
    mae_metric = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_loss = float('inf')

    for epoch in range(args.train_epochs):
        print(f"\nEpoch {epoch}/{args.train_epochs}")
        # 训练一个 epoch
        train_model(args, epoch, model, train_loader, optimizer, criterion, device)

        val_loss, mae_loss = evaluate_model(args, model, vali_loader, criterion, mae_metric, device)

        if val_loss < best_loss:
            best_loss = val_loss
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, f'FLM_form1_epoch{epoch}.pth')
            print(f"Saved best model with loss: {best_loss:.4f}")

        print(f"Epoch {epoch} finished with validation loss: {val_loss:.4f}")






