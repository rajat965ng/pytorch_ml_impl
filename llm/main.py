import datetime
import os
import sys

# pytorch
import torch

# tokenizer
import sentencepiece as spm
from torch import nn
from tqdm import tqdm

from llm.gpt import GPT

# these just improve performance for Ampere architecture
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True

# Empty GPU cache memory
# torch.cuda.empty_cache()

# Architecture parameters
batch_size = 8
context = 512
embed_size = 384
n_layers = 7
n_heads = 7
BIAS = True

# Hyperparameters
lr = 3e-4
dropout = 0.05
weight_decay = 0.01  # prevent sizes of the network too large. It limit the size of the weights so that you force, you push the model to become more flexible and overfit less.
grad_clip = 1.0

# Training parameters
train_iters = 250
eval_interval = 50
eval_iter = 10
compile = False

load_pretrained = True
checkpoint_dir = "models"
checkpoint_fn = "latest.pt"
checkpoint_load_fn = "latest.pt"
dtype = torch.bfloat16

# Mode
inference = False
split = "train"

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Logging
wandb_log = True
wandb_project = "llm1"
wandb_run_name = "llm1-" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

if __name__ == "__main__":
    print(device)
    with open("wiki.txt", "r", encoding="utf-8") as f:
        text = f.read()
    print(text[30000:30300])

    sp = spm.SentencePieceProcessor(model_file="wiki_tokenizer.model")
    vocab_size = sp.piece_size()
    print(f"Tokenizer vocab size: {vocab_size}")

    encode = lambda s: sp.Encode(s)
    decode = lambda l: sp.Decode(l)

    data = encode("Once upon a time")
    print(data)
    print(decode(data))

    if os.path.exists("encoded_data.pt"):
        print("Loading Encoding..")
        data = torch.load('encoded_data.pt')
    else:
        data = torch.tensor(encode(text), dtype=torch.long)
        torch.save(data, 'encoded_data.pt')

    data_size = len(data)
    splt = (int)(0.9 * data_size)
    train_data = data[:splt]
    validation_data = data[splt:]

    print(
        f"Total data: {data_size / 1e6:.2f} Mn | Training Data: {len(train_data) / 1e6:.2f} Mn | Validation data: {len(validation_data) / 1e6:.2f} Mn")


    def get_batch(split):
        data = train_data if split == 'train' else validation_data
        indx = torch.randint(len(data) - context, (batch_size,))
        x = torch.stack([data[i:i + context] for i in indx])
        y = torch.stack([data[i + 1:i + context + 1] for i in indx])
        return x, y

    # Experiment
    # x, y = get_batch('train')
    # print(f'x: {x.shape}, y: {y.shape}')
    # print(x[0][:10])
    # print(y[0][:10])

    model = GPT(vocab_size, n_heads, embed_size, context, n_layers, BIAS, dropout)
    model = model.to(dtype)

    if compile:
        print("Torch :: Compiling Model !!")
        model = torch.compile(model)

    print(sum(p.numel() for p in model.parameters()) / 1e6, " Million Parameters")

    # logits, loss = model(x, y)
    # print(loss.item())


    @torch.no_grad()
    def generate_sample(input):
        t1 = torch.tensor(encode(input), dtype=torch.long)
        t1 = t1[None, :]  # (1,[size of the ids])
        newgen = model.generate(t1, max=64)[0].tolist()
        result = decode(newgen)
        print(f"result: {result}")


    # generate_sample("Once upon a time")


    @torch.no_grad()  # using this decorator we don't pytorch to calculate gradients, backpropagate
    def calculate_loss():
        out = {}  # init dictionary
        model.eval()  # tell architecture that we'll just evaluate and not train
        for split in ['train', 'eval']:  # iterate over both training and evaluation data
            l = torch.zeros(eval_iter)  # init tensor with zeroes. eval_iters will tell how many iterations of
            # calculating loss we'll do for each of these splits
            for i in range(eval_iter):  #
                x, y = get_batch(split)  # obtain batch of training and validation data
                _, loss = model(x, y)  # we just want the loss
                l[i] = loss  # store the loss calculated in i'th interval
            out[split] = l.mean().item()  # after all iterations, we'll take mean of all the losses and store it in
            # the dictionary
        model.train()  # go back to the training mode
        return out  # dictionary of 'train' and 'eval' losses


    print("----------------------Calculating Loss Start-----------------------")
    l = calculate_loss()
    print("----------------------Calculating Loss End-------------------------")
    print(l)
    # if training loss is low but eval loss is high = over-fitting
    # if training loss is near eval loss = good scenario

    # Setting up optimizer
    p_dict = {p_name: p for p_name, p in model.named_parameters() if p.requires_grad}
    weight_decay_p = [p for n, p in p_dict.items() if p.dim() >= 2]
    no_weight_decay_p = [p for n, p in p_dict.items() if p.dim() < 2]

    optimizer_groups = [
        {'params': weight_decay_p, 'weight_decay': weight_decay},
        {'params': no_weight_decay_p, 'weight_decay': 0.0},
    ]
    # to perform gradient decent, tweak and update weights according to diff algos.
    # lr - speed at which you are updating your parameters as you do gradient decent
    # betas - controls the exponential moving average of the gradient and it's square
    optimizer = torch.optim.AdamW(optimizer_groups, lr=lr, betas=(0.9, 0.99))

    # create a scheduler that will change the learning rate of the model during the training schedule
    # CosineAnnealingLR - use cosine fn to do a soft decent of the learning rate.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, train_iters, eta_min=lr / 10)

    start_iteration = 0
    best_validation_loss = float('inf')


    # loading checkpoints
    def load_checkpoint(path):
        print("LLM - Loading Model")
        checkpoint = torch.load(path, map_location=torch.device(device))  # path of the model in subfolder 'models'
        print("-----------------------------checkpoint----------------------------")
        print(checkpoint.keys())
        model.load_state_dict(checkpoint['model_state_dict'])  # dictionary that contains info about weights of the
        # network
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # contains all the relevant values and info.
        # about the optimizer
        iteration = checkpoint['iteration']  # on what iteration we were when we saved the checkpoint, so we continue
        # the training from the same iteration.
        loss = checkpoint['loss']  # current loss when we saved the checkpoint
        print(f"Loaded iteration {iteration} with current loss {loss}")
        return iteration, loss


    if os.path.exists(f"{checkpoint_dir}/{checkpoint_fn}") and load_pretrained:
        start_iteration, loss = load_checkpoint(checkpoint_dir + '/' + checkpoint_load_fn)

    # Inference
    if inference:
        model.eval()
        while True:
            qs = input("Enter text (q to quit):")
            if qs == "":
                continue
            if qs == "q":
                break
            generate_sample(qs)
        sys.exit()

    try:
        print(f"Start training with start_iteration {start_iteration} and train_iters {train_iters}")
        # training loop
        for i in tqdm(range(start_iteration, train_iters)):
            xb, yb = get_batch('train')
            logits, loss = model(xb, yb)

            # Evaluating Loss
            if i % eval_interval == 0 or i == train_iters - 1:
                l = calculate_loss()
                print(f"\n{i}: Train Loss: {l['train']} / validation loss {l['eval']}")
                generate_sample("Once upon a time")
                print(f"Checkpoint condition: best_validation_loss {best_validation_loss} / validation loss {l['eval']}")
                if l['eval'] < best_validation_loss:
                    best_validation_loss = l['eval']
                    print("[CHECKPOINT]: Saving with loss: ", best_validation_loss)
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_validation_loss,
                        'iteration': i,
                    }, checkpoint_dir + "/" + checkpoint_fn)

            optimizer.zero_grad(set_to_none=True)  # set the gradient to zero
            loss.backward()  # backpropogate the loss through the network to calculate all the intermediate gradients

            # Gradient cliping - To prevent gradient explosion that the values of the gradients go too crazy
            nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=grad_clip)

            # Step and tweak each of the parameters of network by amount influenced by the learning rate and
            # in direction that is influenced by the gradient. Direction by the sign of the gradient and magnitude
            # by the amount of the gradient and the learning rate
            optimizer.step()
            # make learning rate change to its next value
            scheduler.step()

            # [+ve gradient] -> you inc. weights -> inc. loss
            # [weight-gradient -> dec. weights -> dec. loss]
            # [-ve gradient] -> you inc. weights -> dec. loss
            # [weight - (-gradient)] -> inc. weights -> dec. loss

    except KeyboardInterrupt:
        print("Training Interrupted. Cleaning Up....")
    finally:
        # Release GPU/CPU memory
        torch.cuda.empty_cache()
        print("GPU memory released..")
        sys.exit(0)