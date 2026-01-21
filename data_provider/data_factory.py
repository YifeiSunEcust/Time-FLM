import pandas as pd
from FLM_Com_dataloader import FLM_Dataset
from torch.utils.data import DataLoader

data_dict = {
    'FLM': FLM_Dataset
}


# 数据提供器函数
def data_provider(args, flag):
    # 根据args.data从data_dict中获取对应的Data类
    Data = data_dict[args.data]

    # 用于保存所有可能的DataLoader
    all_data_loaders = []

    # 遍历seq_len和pred_len的所有组合
    for seq_len in range(args.seq_range[0], args.seq_range[1] + 1):
        for pred_len in range(args.pred_range[0], args.pred_range[1] + 1):

            # 根据flag决定是否打乱和batch_size的配置
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
                size=[seq_len, pred_len],
            )

            # 创建DataLoader
            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                drop_last=drop_last,
                num_workers=args.num_workers,  # 使用的工作线程数
                pin_memory=True  # 启用pin_memory
            )

            # 将当前的data_loader添加到汇总列表
            all_data_loaders.append({
                'seq_len': seq_len,
                'pred_len': pred_len,
                'data_loader': data_loader
            })

    return all_data_loaders

