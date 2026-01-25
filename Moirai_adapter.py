import os
import torch
import numpy as np
import pandas as pd
from data_provider.FLMdata_factory import data_provider
from utils.metrics import metric
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from gluonts.dataset.common import ListDataset
import argparse
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
parser.add_argument('--max_len', type=int, default=100, help='num of encoder layers')
parser.add_argument('--num_patch', type=int, default=40, help='num of encoder layers')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--d_ff', type=int, default=512, help='dimension of fcn')
parser.add_argument('--expert_nums', type=int, default=6, help='num of experts')
parser.add_argument('--top_k', type=int, default=2, help='dimension of expert')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')

parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')

# optimization
parser.add_argument('--batch_size', type=int, default=256, help='batch size of train input data')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
args = parser.parse_args()
# ------------ Existing argparse setup here ------------

if __name__ == "__main__":
    test_loader = data_provider(args, 'test')
    all_inputs = []
    all_truth = []
    for src, tgt, src_mask, tgt_mask in test_loader:
        # src: [B, ctx_len, feat_dim], tgt: [B, pred_len, feat_dim]
        all_inputs.append(src.cpu().numpy())
        all_truth.append(tgt.cpu().numpy())
    inputs = np.concatenate(all_inputs, axis=0)
    truths = np.concatenate(all_truth, axis=0)

    freq = args.freq  # e.g. 'h'
    start = pd.Timestamp("2020-01-01", freq=freq)
    dataset = ListDataset(
        [
            {"start": start, "target": seq.squeeze()}
            for seq in truths
        ],
        freq=freq
    )  # :contentReference[oaicite:5]{index=5}
    MODEL_SIZE = "large"  # or 'small', 'base'
    PRED_LEN = truths.shape[1]
    CTX_LEN = inputs.shape[1]
    model = MoiraiForecast(
        module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-{MODEL_SIZE}"),
        prediction_length=PRED_LEN,
        context_length=CTX_LEN,
        patch_size="auto",
        num_samples=100,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )  # :contentReference[oaicite:6]{index=6} :contentReference[oaicite:7]{index=7}

    predictor = model.create_predictor(batch_size=args.batch_size)  # :contentReference[oaicite:8]{index=8}

    forecasts = predictor.predict(dataset)
    preds = []
    for f in forecasts:
        # f.samples: array (num_samples, prediction_length)
        preds.append(f.mean)
    preds = np.stack(preds, axis=0)

    MAE, MSE, RMSE, MAPE, MSPE, SMAPE, MASE = metric(
        preds.reshape(-1,1), truths.reshape(-1,1)
    )
    print(f"MAE: {MAE:.6f}, MSE: {MSE:.6f}, RMSE: {RMSE:.6f}, MAPE: {MAPE:.6f}")
