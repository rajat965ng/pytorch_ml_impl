import torch.nn.init
from torch import nn
from torch.nn.functional import cross_entropy, softmax

from llm.block import Block


class GPT(nn.Module):
    def __init__(self, vocab_size: int,n_heads, embed_size: int, context: int, n_layers: int, BIAS: bool,dropout):
        super().__init__()
        self.context = context
        ## embedding layer
        self.embeddings = nn.Embedding(vocab_size, embed_size)  ## eg 4096 tokens * 384 factors
        ## positional embeddings layer -
        self.positions = nn.Embedding(context, embed_size)  ## eg 512 positions * 384 factors
        ## layers for all the blocks of the architecture
        ## Blocks will be use to implement multi-head attention mechanism
        ## * is unpacking operator in python. Unpack the elements of the list and pass them as individual arguments to the function
        self.blocks = nn.Sequential(*[Block(n_heads,embed_size,context, BIAS, dropout) for _ in range(n_layers)])
        ## normalization layer - computation
        ## for stabilizing the computations. For cases when the numbers produced during the training become too large/small
        self.ln = nn.LayerNorm(embed_size)
        ## final linear layer - final computation of llm
        ## llm hume end mei probability dega 4096 tokens ki,jo ki next tokens ho sakte hai
        self.final_linear = nn.Linear(embed_size, vocab_size, bias=BIAS)  # eg. 384 * 4096
        self.apply(self._init_weights)  ## jitni bhi layers hai un par init_weights apply kar do

    # Parameter initialization
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0,
                                  std=0.02)  ## layer ko gaussian distribution se init kardiya hai using uske weights
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)  ## bias parameters to zero se init kar denge
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # Inference ke case mei targets dene ki need nahi hai.
    def forward(self, input, targets=None):
        loss = None
        BS, SL = input.shape  # BS x SL
        emb = self.embeddings(input)  # BS x SL x 384 factors
        pos = self.positions(torch.arange(SL))  # SL x 384 factors
        x = emb + pos  # BS x SL x 384
        x = self.blocks(x)  # BS x SL x 384
        x = self.ln(x)  # BS x SL x 384 (embedd size)
        logits = self.final_linear(
            x)  # BS x SL x 4096 (vocab size). Logits are pobabilities in raw formats i.e predictions, means they may not be in between 0 to 1.

        if targets is not None:
            BS, SL, VS = logits.shape  # BS x SL x 4096
            logits = logits.view(BS * SL, VS)
            targets = targets.view(BS * SL)

            # Information is opposite of probability. I = -1 * log(probability)
            # Entropy "avg(info. content (random var))". random var can take multiple values but not in a deterministic way.
            # True distribution Q(o/ps of neural network).
            # Predicted distribution (targets)
            # Difference between true and predicted distribution is cross entropy.
            loss = cross_entropy(logits, targets)

            # Manual Calculation
            counts = logits.exp()
            prob = counts / counts.sum(-1, keepdim=True)
            loss2 = - prob[torch.arange(BS * SL), targets].log().mean()

            if not torch.allclose(loss, loss2):
                print(f"[Loss Diff] Pytorch:{loss.item()} Manual:{loss2.item()}")

        return logits, loss

    # Generate a new sample
    def generate(self, input, max=500):
        for _ in range(max):
            input = input[:,-self.context:]  # (1, input length until max of SL)
            logits, _ = self(input)  # (1,input length,4096)
            logits = logits[:,-1,:] # Pick last prediction, (1,4096)
            probs = softmax(logits,dim=-1) # (1,4096), convert predictions into probabilities
            next = torch.multinomial(probs,1) # collect samples from probability distribution
            input = torch.cat((input,next),dim=1)
        return input
