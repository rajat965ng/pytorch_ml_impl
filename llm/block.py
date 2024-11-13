from torch import nn

from llm.forward_layer import ForwardLayer
from llm.multihead import Multihead


class Block(nn.Module):
    # n_heads is for multi-head attention mechanism
    def __init__(self, n_heads, embed_size, context, bias, dropout):
        super().__init__()
        # The floor division operator // performs division and returns the largest integer that is less than or
        # equal to the division result. It truncates (rounds down) the fractional part of the result, ensuring
        # that the result is always an integer.
        # Each block of transformer has BSxSLxES (embed size).
        # You want to split embedding among all the heads of multi-head attention mechanism.
        head_size = embed_size // n_heads
        # In each block we will do communication and computation.
        # MHA is about communication -  understanding relationships between different tokens of sequence
        self.ma = Multihead(n_heads, head_size, embed_size, context=context, bias=bias, dropout=dropout)
        # feed forward layer is about holding series of computations to increase the complexity
        # of the capabilities of the network
        self.feed_forward = ForwardLayer(embed_size, bias=bias, dropout=dropout)
        # Normalize the inputs across the features for each data point independently.
        # It will subtract the mean of those points and divide it by std. deviation.
        # Followed by some scaling and shifting with some parameters. These parameters are trainable.
        # This will keep those numerical values in a comfortable range of
        # that network.
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)

    # what is the point of having residual connections? as we create networks that are deeper, they can become
    # unstable and they can have problems like vanishing gradients (when gradient becomes very small, the learning
    # stops). By having residual connections, the gradient can travel very fast back through the network, preventing
    # this problem of vanishing.
    def forward(self, x):
        x = x + self.ma(self.ln1(x))  # A residual connection
        x = self.feed_forward(self.ln2(x))  # another residual connection
        return x
