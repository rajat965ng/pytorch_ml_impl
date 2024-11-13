from torch import nn


# Helping the network to create complexity in the mapping that is learning between input and output. We pass input to
# this layer expanding it into way more factors into a higher dimensionality (6 times higher), then we pass into
# non-linearity and then we go back down again to the previous dimensionality and then we do some regularization
class ForwardLayer(nn.Module):
    def __init__(self, embed_size, bias: bool, dropout):
        super().__init__()
        # This mini network will combine different computation layers
        self.network = nn.Sequential(
            nn.Linear(embed_size, 6 * embed_size, bias=bias),
            # Apply a non-linearity. non-linearity function like RELU (rectified linear unit), introduces changes of
            # direction non-linearities in the computations this increases the complexity of the things that you can
            # model with your architecture. RELU introduce sudden change in direction and whatever is less than zero
            # will be set to zero. GELU (Gaussian error linear using) is like RELU but it has a little curve on
            # negative part, that it doesn't set every negative to zero, zero, zero. There are still going to be
            # values that are very small but -ve. This prevents switching off because, if the value of neuron is -ve
            # the RELU fn will completely de-activate the neuron.
            nn.GELU(),
            nn.Linear(6 * embed_size, embed_size, bias=bias),
            # Regularization mechanism to prevent the model from over-fitting. Deactivate some of the neurons randomly.
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.network(x)
        return x
