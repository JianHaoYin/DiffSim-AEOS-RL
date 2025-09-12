import torch
import torch.nn as nn

class ScaleDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaleDotProductAttention, self).__init__()
        self.d_k = d_k

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, attn_mask=None):
        """
        Args:
            d_k is same as d_tensor
            Q: [batch_size, n_heads, length, d_tensor]
            K: [batch_size, n_heads, length, d_tensor]
            V: [batch_size, n_heads, length, d_tensor]
            attn_mask: [batch_size, n_heads, length, length]
        Returns:
            context: [batch_size, n_heads, length, d_tensor]
            attn: [batch_size, n_heads, length, length]
        """
        scores = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.d_k).float())
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)

        attn = self.softmax(scores)
        context = torch.matmul(attn, V)

        return context, attn