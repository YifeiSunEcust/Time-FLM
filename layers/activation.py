import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        """
        SwiGLU activation function module.
        Args:
            input_dim (int): Input feature dimension.
            hidden_dim (int): Hidden dimension for intermediate representation.
        """
        super(SwiGLU, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim * 2)  # Double output for gating
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        """
        Forward pass for SwiGLU.
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, input_dim).
        Returns:
            Tensor: Output tensor of the same shape as input.
        """
        x = self.fc1(x)  # Shape: (batch_size, seq_len, hidden_dim * 2)
        x1, x2 = torch.chunk(x, chunks=2, dim=-1)  # Split into two halves for gating
        x = F.silu(x1) * x2  # Apply SiLU (Swish) and gating
        x = self.fc2(x)  # Shape: (batch_size, seq_len, input_dim)
        return x

