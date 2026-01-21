import torch
import torch.nn as nn
import torch.nn.functional as F

class ReplicationPad1d(nn.Module):
    def __init__(self, padding) -> None:
        super(ReplicationPad1d, self).__init__()
        self.padding = padding

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        replicate_padding = input[:, :, -1].unsqueeze(-1).repeat(1, 1, self.padding[-1])
        output = torch.cat([input, replicate_padding], dim=-1)
        return output


class TokenEmbedding(nn.Module):
    def __init__(self, patch_len, d_model):
        super(TokenEmbedding, self).__init__()
        self.patch_len = patch_len
        self.d_model = d_model
        # Linear projection to the model dimension
        self.proj = nn.Linear(patch_len, d_model)

    def forward(self, x):
        return self.proj(x)

class PatchEmbedding(nn.Module):
    def __init__(self, d_model, num_patches, stride, dropout):
        super(PatchEmbedding, self).__init__()
        self.num_patches = num_patches  # Target number of patches per sequence
        self.stride = stride
        self.padding_patch_layer = ReplicationPad1d((0, stride))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = TokenEmbedding(1, d_model)  # Temp size 1 until dynamically calculated

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, n_vars, seq_len = x.shape
        # Calculate the dynamic patch size based on seq_len and num_patches
        seq_len_tensor = torch.tensor(seq_len, dtype=torch.float)  # Ensure seq_len is a tensor
        patch_size = torch.ceil(seq_len_tensor / self.num_patches).long()  # Patch size = seq_len / num_patches

        # Pad the input sequence to match the required length for patching
        x = self.padding_patch_layer(x)

        # Unfold the sequence dynamically with the calculated patch_size and stride
        x = x.unfold(dimension=-1, size=patch_size.item(), step=self.stride)  # Use patch_size.item() for unfold
        x = torch.reshape(x, (batch_size * n_vars, x.shape[2], x.shape[3]))  # Flatten for embedding
        # x shape: [batch_size * n_vars, num_patches, patch_size]
        # Apply token embedding to the patches
        x = self.value_embedding(x)

        return self.dropout(x), n_vars

# Example usage
batch_size = 2
seq_len = 10  # Varying sequence length for each input
num_patches = 3  # Target number of patches

# Create a random tensor of shape [batch_size, 1, seq_len] as an example input
x = torch.randn(batch_size, 1, seq_len)

# Create the PatchEmbedding layer
patch_embedder = PatchEmbedding(d_model=16, num_patches=num_patches, stride=2, dropout=0.1)

# Pass the input through the patch embedding layer
output, n_vars = patch_embedder(x)

# Output shape: [batch_size * n_vars, num_patches, d_model]
print(output.shape)  # Expected output shape [batch_size * n_vars, num_patches, d_model]
