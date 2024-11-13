import torch
from torch import nn

from llm.head import Head


class Multihead(nn.Module):
    def __init__(self, n_heads, head_size, embed_size,context, bias: bool, dropout):
        super().__init__()
        # Create a list of heads. Head is where all these computations are gonna learn about the relationships
        # between different tokens.
        self.heads = nn.ModuleList([Head(head_size,embed_size,context=context,bias=bias,dropout=dropout) for _ in range(n_heads)])
        # after all the heads done processing, we will combine them and projecting it back on embedding size (384)
        self.combine = nn.Linear(head_size * n_heads, embed_size, bias=bias)  # 54 * 7 ~ 378
        # some regularization to prevent over-fitting
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # concatenate the result of passing the input through all the heads
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        # Each head outputs (BS, SL, head_size)
        x = self.combine(x)  # projected to BS x SL x 384
        x = self.dropout(x)
        return x
