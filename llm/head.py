import torch
from torch import nn
from torch.nn.functional import softmax


# what is the difference between two tokens in a sequence at an abstract level
class Head(nn.Module):
    def __init__(self, head_size, embed_size, bias, context, dropout):
        super().__init__()
        # queries,keys and values are the projections of the embeddings of the tokens of our sequences.
        # info -> MHA -> split(embeddings(each tokens(of sequences))) -> between diff. heads.
        # queries & keys are to find compatibility score between the tokens of the sequence.
        # How aligned is the embedding value of a token with embedding values of other tokens.
        self.queries = nn.Linear(embed_size, head_size, bias=bias)  # 384 x 54 dimensions
        # keys showcase the features of each of the other tokens
        self.keys = nn.Linear(embed_size, head_size, bias=bias)
        # projections of content of our embeddings
        self.values = nn.Linear(embed_size, head_size, bias=bias)

        # parameter of class not going to be included in the calculations of the gradients.
        # its a square matrix. Lower left triangle full of ones. Upper right triangle full of zeroes.
        self.register_buffer('tril', torch.tril(torch.ones(context, context)))  # 512 x 512
        # Regularization layer to keep our computation save from overfitting
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        BS, SL, VS = x.shape
        q = self.queries(x)  # BS x SL x 54
        k = self.keys(x)  # BS x SL x 54
        v = self.values(x)  # BS x SL x 54

        # attention score or attention matrix or attention weights = mat multiplication between q & k
        # 512 x 54 @ 512 x 54 -> 512 x 54 @ 54 x 512
        # also we need to normalize this multiplication as the numbers are getting very large.
        # So we divide this by sqrt(last dimension of k)
        attn_w = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # BS, SL x SL
        # attn_w -> 512 x 512, 1st row -> degree of alignment of 1st token with each of the other token in sequence
        # if the dot product is high +ve then the tokens are highly aligned
        # if they are high -ve the tokens are anti-aligned
        # if product is 0, then the tokens are perpendicular

        # It's very important that every token must pay attention to the past not the future We will update attn_w
        # matrix to mask future weights softmax lagakar hum probabilities nikalenge. But hum chahte hai ki jaha par
        # zero hai usko softmax read na kare isliye zeroes ko infinity bana do.
        attn_w = attn_w.masked_fill(self.tril[:SL, :SL] == 0, float('-inf'))
        # softmax lagate hue last dimension par focus rakhna hai
        attn_w = softmax(attn_w,dim=-1) # BS, SL, SL

        # attn_w - degree of compatibility of 1st token with other token (1st row)
        # v - value of the first factor of embeddings for all of the tokens
        x = attn_w @ v # BS, SL, 54

        return x
