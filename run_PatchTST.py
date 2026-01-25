from data_provider.FLM_Com_datafactory import data_provider
from models import Autoformer, DLinear, TimeLLM, TimesNet, PatchTST, Informer, iTransformer, \
    TimeXer, TiDE, MtsLLM, woLLM, TLinear
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
import matplotlib.pyplot as plt
os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description='Time-LLM')
# basic config
parser.add_argument('--model', type=str, default='TiDE',
                    help='model name, options: [TimeLLM, Autoformer, DLinear, TLinear, TimesNet, PatchTST, Informer, iTransformer'
                         'TimeXer, TiDE, MtsLLM, woLLM]')
parser.add_argument('--task_name', type=str, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')

# data loader
parser.add_argument('--data', type=str, default='EFP_each', help='dataset type')
parser.add_argument('--root_path', type=str, default='F:\FLM\datasets\PatchTST_datasets', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='EFP_long.csv', help='data file')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, '
                         'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                         'you can also use more detailed freq like 15min or 3h')
parser.add_argument('--features', type=str, default='MS',help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S: univariate predict univariate, MS:multivariate predict univariate')
# model define
parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
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
parser.add_argument('--seq_len', type=int, default=24, help='input sequence length')
parser.add_argument('--pred_len', type=int, default=8, help='prediction sequence length')
parser.add_argument('--label_len', type=int, default=0, help='label length')

# optimization
parser.add_argument('--batch_size', type=int, default=256, help='batch size of train input data')
parser.add_argument('--train_epochs', type=int, default=3, help='train epochs')
parser.add_argument('--num_experiments', type=int, default=1, help='Number of training sessions')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='optimizer learning rate')
args = parser.parse_args()


def main_run(model, optimizer, train_loader, vali_loader, test_loader, num_experiments):
    time_now = t.time()
    train_steps = len(train_loader)
    criterion = nn.MSELoss()
    mae_metric = nn.L1Loss()
    train_loss = []
    t0 = t.time()
    for epoch in range(args.train_epochs):
        iter_count = 0
        loss_count = 0
        model.train()
    # torch.cuda.empty_cache()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader)):
            iter_count += 1
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

        # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

            with torch.cuda.amp.autocast():
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                outputs = outputs[:, -args.pred_len:, :]
                batch_y = batch_y[:, -args.pred_len:, :]
                loss = criterion(outputs, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_count += loss.item()

            if (i + 1) % 100 == 0:
                print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (t.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = t.time()

        train_loss.append(loss_count / i)
        vali_loss, vali_mae_loss = validate(args, model, vali_loader, criterion, mae_metric)
        print("Epoch: {0} | Vali Loss: {1:.7f} MAE Loss: {2:.7f}".format(epoch + 1, vali_loss, vali_mae_loss))

    t1 = t.time()
    train_time = t1 - t0

    predict = []
    truth = []
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader)):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

        # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
            with torch.cuda.amp.autocast():
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs[:, -args.pred_len:, :]
                batch_y = batch_y[:, -args.pred_len:, :]

            pred = outputs.detach()
            true = batch_y.detach()
            predict.append(pred.cpu().numpy())
            truth.append(true.cpu().numpy())
    # 测试集正确率
    pre_total = np.concatenate(predict, axis=0)  # 将所有批次的预测值拼接起来
    true_total = np.concatenate(truth, axis=0)  # 将所有批次的真实值拼接起来

    pre_total = pre_total.reshape(-1, 1)
    true_total = true_total.reshape(-1, 1)
    df = pd.DataFrame({
        'Prediction': pre_total.flatten(),  # 确保是1维
        'GroundTruth': true_total.flatten()
    })

    # 保存为CSV
    df.to_csv(f'{args.model}.csv', index=False)

    MAE, MSE, RMSE, MAPE, MSPE, SMAPE, MASE = metric(pre_total, true_total)


    print(MAE)
    print(MSE)
    print(RMSE)
    print(MAPE)
    print(MSPE)
    print(SMAPE)
    print(MASE)
    print(train_time)
    return MAE, MSE, RMSE, MAPE, MSPE, SMAPE, MASE, train_time


if __name__ == "__main__":
    seed = random.randint(0, 10000)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_loader = data_provider(args, 'train')
    vali_loader = data_provider(args, 'val')
    test_loader = data_provider(args, 'test')
    results = {
        "MAE": [],
        "MSE": [],
        "RMSE": [],
        "MAPE": [],
        "MSPE": [],
        "SMAPE": [],
        "MASE": [],
        "Time": []
    }
    for i in range(args.num_experiments):
        print(f"Running experiment {i + 1}")
        # 假设 main_run() 返回一个包含多个指标的元组
        if args.model == 'TimeLLM':
            model = TimeLLM.Model(args).float().to(device)
        elif args.model == 'DLinear':
            model = DLinear.Model(args).float().to(device)
        elif args.model == 'Autoformer':
            model = Autoformer.Model(args).float().to(device)
        elif args.model == 'TimesNet':
            model = TimesNet.Model(args).float().to(device)
        elif args.model == 'PatchTST':
            model = PatchTST.Model(args).float().to(device)
        elif args.model == 'Informer':
            model = Informer.Model(args).float().to(device)
        elif args.model == 'iTransformer':
            model = iTransformer.Model(args).float().to(device)
        elif args.model == 'TimeXer':
            model = TimeXer.Model(args).float().to(device)
        elif args.model == 'TiDE':
            model = TiDE.Model(args).float().to(device)
        elif args.model == 'MtsLLM':
            model = MtsLLM.Model(args).float().to(device)
        elif args.model == 'woLLM':
            model = woLLM.Model(args).float().to(device)
        elif args.model == 'TLinear':
            model = TLinear.Model(args).float().to(device)

        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
        mae, mse, rmse, mape, mspe, smape, mase, time = main_run(model, optimizer, train_loader, vali_loader, test_loader, i)

        # 使用循环来添加结果
        for key, value in zip(results.keys(), [mae, mse, rmse, mape, mspe, smape, mase, time]):
            results[key].append(value)
        print(f'Saved {i + 1}')

    print("All experiment results:", results["MAE"])

    # 将结果保存为 CSV 文件
    df = pd.DataFrame(results).T
    df.to_csv(f"./results/exp/{args.model}_EFPeach_{args.seq_len}-{args.label_len}-{args.pred_len}.csv", index=False)
    print("Results saved to csv")