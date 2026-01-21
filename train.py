import torch
import numpy as np
from tqdm import tqdm
import time as t
import torch.nn as nn


def train_model(args, epoch, model, train_loader, optimizer, criterion, device):
    model.train()
    time_now = t.time()
    train_steps = len(train_loader)
    total_loss = 0.0
    iter_count = 0
    Huber_loss = nn.HuberLoss(delta=1.0)
    for i, (src, tgt, src_mask, tgt_mask) in tqdm(enumerate(train_loader)):
        src, tgt, src_mask, tgt_mask = src.to(device), tgt.to(device), src_mask.to(device), tgt_mask.to(device)
        pre_len = tgt.size(1)
        iter_count += 1
        with torch.amp.autocast('cuda'):
            optimizer.zero_grad()
            outputs = model(src, src_mask)
            outputs = outputs[:, -pre_len:, :]
            tgt_mask = tgt_mask.bool()
            loss = criterion(outputs[tgt_mask], tgt[tgt_mask])+0.5*Huber_loss(outputs[tgt_mask], tgt[tgt_mask])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (i + 1) % 100 == 0:
            print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
            speed = (t.time() - time_now) / iter_count
            left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
            print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
            iter_count = 0
            time_now = t.time()

    avg_loss = total_loss / len(train_loader)
    print(f"Average Loss for this epoch: {avg_loss:.4f}")

