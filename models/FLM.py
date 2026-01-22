import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding_wo_time


class BasicExpert(nn.Module):
    def __init__(self, feature_in, feature_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_in, feature_in),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_in, feature_out)
        )

    def forward(self, x):
        return self.net(x)


class MOERouter(nn.Module):
    def __init__(self, hidden_dim, expert_number, top_k):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, expert_number)
        self.expert_number = expert_number
        self.top_k = top_k

    def forward(self, hidden_states):
        router_logits = self.gate(hidden_states)  # shape is (b * s, expert_number)
        routing_probs = F.softmax(router_logits, dim=-1, dtype=torch.float)

        router_weights, selected_experts = torch.topk(
            routing_probs, self.top_k, dim=-1
        ) 

        router_weights = router_weights / router_weights.sum(dim=-1, keepdim=True)
        router_weights = router_weights.to(hidden_states.dtype)

        expert_mask = F.one_hot(
            selected_experts,
            num_classes=self.expert_number
        )  
        expert_mask = expert_mask.permute(2, 1, 0) 

        return router_logits, router_weights, selected_experts, expert_mask


class SparseMOE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.d_model
        self.expert_number = config.expert_number
        self.top_k = config.top_k
        self.experts = nn.ModuleList(
            [
                BasicExpert(self.hidden_dim, self.hidden_dim) for _ in range(self.expert_number)
            ]
        )

        self.router = MOERouter(self.hidden_dim, self.expert_number, self.top_k)

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.size()
        hidden_states = x.view(-1, hidden_dim) 
        router_logits, router_weights, selected_experts_indices, expert_mask = self.router(hidden_states)

        final_hidden_states = torch.zeros(
            (batch_size * seq_len, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )

        for expert_idx in range(self.expert_number):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            current_state = hidden_states.unsqueeze(
                0
            )[:, top_x, :].reshape(-1, hidden_dim) 
            current_hidden_states = expert_layer(
                current_state
            ) * router_weights[top_x, idx].unsqueeze(-1)  

            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        final_hidden_states = final_hidden_states.reshape(batch_size, seq_len, hidden_dim)

        return final_hidden_states, router_logits 


class ShareExpertMOE(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.moe_model = SparseMOE(config)
        self.shared_experts = nn.ModuleList(
            [
                BasicExpert(
                    config.d_model, config.d_model
                ) for _ in range(config.shared_experts_number)
            ]
        )

    def forward(self, x):
        sparse_moe_out, router_logits = self.moe_model(x)
        shared_experts_out = [
            expert(x) for expert in self.shared_experts
        ] 

        shared_experts_out = torch.stack(
            shared_experts_out, dim=0
        ).sum(dim=0)
        return sparse_moe_out + shared_experts_out, router_logits

class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(config.d_model, config.n_heads, dropout=config.dropout)
        self.moe = ShareExpertMOE(config)
        self.dropout = nn.Dropout(config.dropout)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)

    def forward(self, src, src_key_padding_mask=None):
        x = src.permute(1, 0, 2)
        src2 = self.self_attn(x, x, x, key_padding_mask=src_key_padding_mask)[0]
        src2 = src2.permute(1, 0, 2)
        src = src + self.dropout(src2)
        src = self.norm1(src)
        
        src2, _ = self.moe(src)
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.embed = DataEmbedding_wo_time(c_in=1, d_model=config.d_model, dropout=config.dropout)
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.e_layers)])
        self.proj = nn.Linear(config.d_model, 1)

    def forward(self, src, src_key_padding_mask=None):
        src = self.embed(src)
        output = src
        for layer in self.layers:
            output = layer(output, src_key_padding_mask=src_key_padding_mask)
        output = self.proj(output)
        return output

