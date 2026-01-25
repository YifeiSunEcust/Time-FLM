import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import TimeFLMEmbedding


# ===================== Expert =====================
class Expert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        return self.net(x)


# ===================== MoE Layer =====================
class MoELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.expert_nums
        self.top_k = config.top_k

        self.experts = nn.ModuleList(
            [Expert(config) for _ in range(self.num_experts)]
        )
        self.gate = nn.Linear(config.d_model, self.num_experts)

    def forward(self, x, TimeStage):
        # -------- Router (explicit conditioning) --------
        router_input = x + TimeStage.detach()
        gate_score = F.softmax(self.gate(router_input), dim=-1)  # (B, T, E)

        topk_val, topk_idx = torch.topk(gate_score, self.top_k, dim=-1)

        output = torch.zeros_like(x)

        # -------- Sparse expert execution --------
        for eid, expert in enumerate(self.experts):
            mask = (topk_idx == eid)
            weight = (mask.float() * topk_val).sum(dim=-1)  # (B, T)
            selected = weight > 0

            if selected.any():
                x_sel = x[selected]                  # (N, d_model)
                y_sel = expert(x_sel)                # (N, d_model)
                output[selected] += y_sel * weight[selected].unsqueeze(-1)

        return output


# ===================== Transformer Encoder Layer =====================
class TransformerEncoderLayerWithMoE(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            config.d_model,
            config.n_heads,
            dropout=config.dropout,
            batch_first=False
        )

        self.moe = MoELayer(config)

        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)

        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

    def forward(self, src, TimeStage, src_mask=None, src_key_padding_mask=None, is_causal=False):
        attn_out, _ = self.self_attn(
            src,
            src,
            src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            is_causal=is_causal
        )
        src = self.norm1(src + self.dropout1(attn_out))

        moe_out = self.moe(
            src.permute(1, 0, 2),          # (B, T, d)
            TimeStage.permute(1, 0, 2)           # (B, T, d)
        ).permute(1, 0, 2)

        src = self.norm2(src + self.dropout2(moe_out))
        return src


# ===================== Model =====================
class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embedding = TimeFLMEmbedding(
            c_in=1,
            d_model=config.d_model,
            dropout=config.dropout
        )

        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderLayerWithMoE(config) for _ in range(config.e_layers)]
        )

        self.proj = nn.Linear(config.d_model, 1)

    def forward(self, x, x_mask):
        """
        x      : (B, T, 1)
        x_mask : (B, T)
        """
        # -------- Embedding --------
        x_embed, TimeStage = self.embedding(x)      # BOTH preserved

        # (T, B, d)
        src = x_embed.permute(1, 0, 2)
        TimeStage = TimeStage.permute(1, 0, 2)

        # -------- Encoder Stack --------
        for layer in self.encoder_layers:
            src = layer(
                src,
                TimeStage,
                src_key_padding_mask=x_mask
            )

        # -------- Output --------
        src = src.permute(1, 0, 2)
        return self.proj(src)
