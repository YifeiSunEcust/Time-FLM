import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils import weight_norm
import math


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000, precision=torch.half, learnable=False):
        super().__init__()
        # 计算 \theta_i
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        inv_freq = inv_freq.half()

        self.learnable = learnable
        if learnable:
            self.inv_freq = torch.nn.Parameter(inv_freq)
            self.max_seq_len_cached = None
        else:
            self.register_buffer('inv_freq', inv_freq)
            self.max_seq_len_cached = None
            self.cos_cached = None
            self.sin_cached = None
        self.precision = precision

    def forward(self, x, seq_dim=1, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[seq_dim]
        if self.max_seq_len_cached is None or (seq_len > self.max_seq_len_cached):
            self.max_seq_len_cached = None if self.learnable else seq_len
            # 生成 token 序列索引 t = [0, 1,..., seq_len-1]
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            # 对应m * \theta
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            # 将 m * \theta 拼接两次，对应复数的实部和虚部
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            if self.precision == torch.bfloat16:
                emb = emb.float()

            # [sx, 1 (b * np), hn]
            cos_cached = emb.cos()[:, None, :]  # 计算得到cos(m*\theta)
            sin_cached = emb.sin()[:, None, :]  # 计算得到cos(m*\theta)
            if self.precision == torch.bfloat16:
                cos_cached = cos_cached.bfloat16()
                sin_cached = sin_cached.bfloat16()
            if self.learnable:
                return cos_cached, sin_cached
            self.cos_cached, self.sin_cached = cos_cached, sin_cached
        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]

    def _apply(self, fn):
        if self.cos_cached is not None:
            self.cos_cached = fn(self.cos_cached)
        if self.sin_cached is not None:
            self.sin_cached = fn(self.sin_cached)
        return super()._apply(fn)


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=x1.ndim - 1)


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        hour_size = 24
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding

        self.hour_embed = Embed(hour_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        hour_x = self.hour_embed(x[:, :, 3])
        day_x = self.day_embed(x[:, :, 2])
        month_x = self.month_embed(x[:, :, 1])

        return hour_x + day_x + month_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 3, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class ReplicationPad1d(nn.Module):
    def __init__(self, padding) -> None:
        super(ReplicationPad1d, self).__init__()
        self.padding = padding

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        replicate_padding = input[:, :, -1].unsqueeze(-1).repeat(1, 1, self.padding[-1])
        output = torch.cat([input, replicate_padding], dim=-1)
        return output


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = ReplicationPad1d((0, stride))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = TokenEmbedding(patch_len, d_model)

        # Positional embedding
        # self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x)
        return self.dropout(x), n_vars


class DataEmbedding_wo_time(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_time, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


class TimeStageEmbedding(nn.Module):
    """
    对连续值的 stage 信息（范围为 0~1）进行绝对位置编码
    """
    def __init__(self, d_model):
        super(TimeStageEmbedding, self).__init__()
        self.d_model = d_model

    def forward(self, x):

        batch_size, seq_len, _ = x.shape
        position = x.squeeze(-1)  # 移除最后一维，变成 [batch_size, seq_len]

        # 构建 div_term，形状为 [d_model // 2]
        div_term = (torch.arange(0, self.d_model, 2, device=x.device).float()
                    * -(math.log(10000.0) / self.d_model)).exp()

        pos_enc = torch.zeros(batch_size, seq_len, self.d_model, device=x.device)
        pos_enc[:, :, 0::2] = torch.sin(position.unsqueeze(-1) * div_term)
        pos_enc[:, :, 1::2] = torch.cos(position.unsqueeze(-1) * div_term)

        return pos_enc


class input_PatchEmbedding(nn.Module):
    def __init__(self, d_model, num_patch, dropout):
        super(input_PatchEmbedding, self).__init__()
        # Patching
        self.num_patch = num_patch
        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = TokenEmbedding(1, d_model)
        # Positional embeddingRotaryEmbedding
        self.RoPE = RotaryEmbedding(dim=d_model)
        # TimeStage embedding
        self.TimeStage_embedding = TimeStageEmbedding(d_model=d_model)
        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def ReplicationPad1d(self, x, padding):
        replicate_padding = x[:, :, -1].unsqueeze(-1).repeat(1, 1, padding[-1])
        output = torch.cat([x, replicate_padding], dim=-1)
        return output

    def forward(self, x, x_mark):
        # do patching
        batch_size, seq_len, n_vars = x.shape[0], x.shape[1], x.shape[2]
        patch_len = (seq_len + self.num_patch - 1) // self.num_patch
        padded_seq_len = patch_len * self.num_patch
        padding = padded_seq_len - seq_len

        x = x.permute(0, 2, 1)
        x = self.ReplicationPad1d(x=x, padding=(0, padding))
        x = x.unfold(dimension=-1, size=patch_len, step=patch_len)
        x = torch.mean(x, dim=-1)  # 对patch len维度取平均
        x = x.permute(0, 2, 1)

        x_mark = x_mark.permute(0, 2, 1)
        x_mark = self.ReplicationPad1d(x=x_mark, padding=(0, padding))
        x_mark = x_mark.unfold(dimension=-1, size=patch_len, step=patch_len)
        x_mark = torch.mean(x_mark, dim=-1)  # 对patch len维度取平均
        x_mark = x_mark.permute(0, 2, 1)

        # Input encoding
        x_value = self.value_embedding(x)

        cos_emb, sin_emb = self.RoPE(x)
        cos_emb, sin_emb = cos_emb.permute(1, 0, 2), sin_emb.permute(1, 0, 2)
        x_rotated = x * cos_emb + rotate_half(x) * sin_emb

        x_timestage = self.TimeStage_embedding(x_mark)

        x = x_value + x_rotated

        return self.dropout(x), x_timestage


class output_PatchProjection(nn.Module):
    def __init__(self, d_model, num_patch, dropout):
        super(output_PatchProjection, self).__init__()
        # Patching
        self.num_patch = num_patch
        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = TokenEmbedding(d_model, 1)
        # Positional embedding
        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def ReplicationPad1d(self, x, padding):
        replicate_padding = x[:, :, -1].unsqueeze(-1).repeat(1, 1, padding[-1])
        output = torch.cat([x, replicate_padding], dim=-1)
        return output

    def forward(self, x, pred_len):
        # do patching
        batch_size, num_patch, d_model = x.shape[0], x.shape[1], x.shape[2]
        patch_len = (num_patch + pred_len - 1) // pred_len
        padded_seq_len = patch_len * pred_len
        padding = padded_seq_len - num_patch
        x = x.permute(0, 2, 1)
        x = self.ReplicationPad1d(x=x, padding=(0, padding))
        x = x.unfold(dimension=-1, size=patch_len, step=patch_len)
        x = torch.mean(x, dim=-1)  # 对patch len维度取平均
        x = x.permute(0, 2, 1)
        # Input encoding
        x = self.value_embedding(x)
        return self.dropout(x)
