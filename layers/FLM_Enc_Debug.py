import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.activation import SwiGLU


class MoELayer(nn.Module):
    def __init__(self, input_dim, num_experts, hidden_dim, dropout, activation):
        """
        MoE layer for replacing the feedforward network in the encoder.
        Args:
            input_dim (int): Input dimension.
            num_experts (int): Number of experts.
            hidden_dim (int): Hidden dimension for each expert.
            dropout (float): Dropout rate.
        """
        super(MoELayer, self).__init__()
        self.num_experts = num_experts
        self.gate = nn.Linear(input_dim, num_experts)
        self.activation = SwiGLU(input_dim=hidden_dim, hidden_dim=hidden_dim) if activation == "SwiGLU" else F.gelu
        self.experts = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                self.activation,
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, input_dim)
            ) for _ in range(num_experts)]
        )

    def forward(self, x):
        # Time Stage Router
        gate_scores = F.softmax(self.gate(x), dim=-1)  # (batch_size, seq_len, num_experts)

        # Compute expert outputs
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2)  # (batch_size, seq_len, num_experts, input_dim)

        # Weighted sum of experts
        output = torch.einsum('bse,bsei->bsi', gate_scores, expert_outputs)  # (batch_size, seq_len, input_dim)
        return output


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, num_experts=4, activation="SwiGLU"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.MoE = MoELayer(input_dim=d_model, num_experts=num_experts, hidden_dim=d_ff, dropout=dropout, activation="SwiGLU")
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.MoE(y))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []

        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns