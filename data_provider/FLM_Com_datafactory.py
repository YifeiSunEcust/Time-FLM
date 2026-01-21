from data_provider.data_loader import FLM_Dataset
from torch.utils.data import DataLoader
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
        seq_len=args.seq_len,
        pred_len=args.pred_len
        )
    # 创建DataLoader
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        drop_last=drop_last,
        num_workers=5,  # 使用的工作线程数
        pin_memory=True  # 启用pin_memory
        )
    return data_loader