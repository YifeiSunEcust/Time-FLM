import torch
from torch import nn
# from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import input_PatchEmbedding, output_PatchProjection
from layers.FLM_Enc import Encoder, EncoderLayer


class Model(nn.Module):
    def __init__(self, d_model=128, num_patch=20, dropout=0.1, factor=1, n_heads=8, d_ff=128, activation='gelu',
                 e_layers=3, num_experts=4):
        super().__init__()
        # patching and embedding
        self.enc_PatchEmbedding = input_PatchEmbedding(d_model, num_patch, dropout)
        self.out_projection = output_PatchProjection(d_model, num_patch, dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(True, factor, attention_dropout=dropout,
                                      output_attention=False), d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    num_experts=num_experts,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.RMSNorm(d_model)
        )

    def forecast(self, x_enc, x_mark_enc, pred_len):
        # x_enc size [batch size, seq_len, 1]
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        x_enc, x_timestage = self.enc_PatchEmbedding(x_enc, x_mark_enc)
        # Encoder
        # enc_out: [batch size, seq_len, d_model]
        enc_out, attns = self.encoder(x_enc, x_timestage)

        dec_out = self.out_projection(enc_out, pred_len)

        return dec_out

    def forward(self, x_enc, x_mark_enc, pred_len, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, pred_len)
        return dec_out


batch_size = 64
seq_len = 100
pred_len = 30
x = torch.randn(batch_size, seq_len, 1)
x_mark = torch.randn(batch_size, seq_len, 1)
print(len(x[1]))

