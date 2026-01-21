from torch.utils.data import DataLoader
from data_provider.FLM_dataset import FLM_Dataset
import numpy as np
import torch


def collate_fn(batch):
    # 计算batch中每个样本的序列长度和预测长度
    max_seq_len = max(len(sample[0]) for sample in batch)
    max_pred_len = max(len(sample[1]) for sample in batch)

    # 初始化填充后的输入和目标列表
    padded_inputs = np.zeros((len(batch), max_seq_len + max_pred_len, batch[0][0].shape[1]), dtype=np.float32)
    padded_targets = np.zeros((len(batch), max_pred_len, batch[0][1].shape[1]), dtype=np.float32)

    # 初始化填充编码
    src_mask = np.zeros((len(batch), max_seq_len + max_pred_len), dtype=np.float32)
    tgt_mask = np.zeros((len(batch), max_pred_len), dtype=np.float32)

    # 遍历batch中的每个样本
    for i, (seq_x, seq_y) in enumerate(batch):
        seq_x_len = len(seq_x)
        seq_y_len = len(seq_y)

        # 复制原始数据到填充数组
        padded_inputs[i, :seq_x_len] = seq_x
        padded_targets[i, :seq_y_len] = seq_y

        # 生成填充掩码：用1表示有效数据，用0表示填充数据
        src_mask[i, :seq_x_len] = 1
        tgt_mask[i, :seq_y_len] = 1

    # 直接转换为 PyTorch tensor，避免单个元素逐个转换
    return torch.from_numpy(padded_inputs), torch.from_numpy(padded_targets), torch.from_numpy(src_mask), torch.from_numpy(tgt_mask)


def data_provider(args, flag):
    # 根据args.data从data_dict中获取对应的Data类
    Data = FLM_Dataset
    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size

    # 创建Dataset实例
    data_set = Data(
        root_path=args.root_path,
        flag=flag,
        total_range=args.total_range
        )
    # 创建DataLoader
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle_flag,
        drop_last=drop_last,
        num_workers=5,  # 使用的工作线程数
        pin_memory=True  # 启用pin_memory
        )
    return data_loader