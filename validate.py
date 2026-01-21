import torch
import numpy as np
from tqdm import tqdm
import time as t


def evaluate_model(args, model, vali_loader, criterion, mae_metric, device):
    """
    在所有 DataLoader 上评估模型性能
    """
    model.eval()  # 设置模型为评估模式
    total_loss = []
    total_mae_loss = []
    with torch.no_grad():
        for i, (src, tgt, src_mask, tgt_mask) in tqdm(enumerate(vali_loader)):
            src, tgt, src_mask, tgt_mask = src.to(device), tgt.to(device), src_mask.to(device), tgt_mask.to(device)
            pre_len = tgt.size(1)
            with torch.amp.autocast('cuda'):
                outputs = model(src, src_mask)
            pred = outputs.detach()
            true = tgt.detach()
            tgt_mask = tgt_mask.bool()
            pred = pred[:, -pre_len:, :]

            loss = criterion(pred[tgt_mask], true[tgt_mask])
            mae_loss = mae_metric(pred[tgt_mask], true[tgt_mask])

            total_loss.append(loss.item())
            total_mae_loss.append(mae_loss.item())

    total_loss = np.average(total_loss)
    total_mae_loss = np.average(total_mae_loss)

    print(f"Average Loss across all DataLoaders: {total_loss:.4f}")
    return total_loss, total_mae_loss