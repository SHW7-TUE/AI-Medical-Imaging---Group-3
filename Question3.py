# =====================
# Imports
# =====================

import os
import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from datasets import load_dataset, concatenate_datasets

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm.auto import tqdm

# =====================
# Reproducibility
# =====================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)


# =====================
# Check GPU/CPU
# =====================

if torch.cuda.is_available():
    device = torch.device("cuda")  # Windows/Linux
    print("gpuuuuuuuuu")

elif torch.backends.mps.is_available():
    device = torch.device("mps")  # MacOS
else:
    device = torch.device("cpu")
    print("cpuuuuu")

# =====================
# Dataset
# =====================

print("Loading dataset...")
ds_dict = load_dataset("MartiHan/Open-MELON-VL-2.5K")
ds_all = concatenate_datasets(list(ds_dict.values()))

captions = [str(x) for x in ds_all["caption"]]
#print("Captions:", len(captions))
#print("Example caption:", captions[0])

SEP = "\n<ENDC>\n"
USE_ENDC = True  # set to False to train without the separator

if USE_ENDC:
    text = SEP.join(captions)
else:
    # no explicit boundary token between captions
    text = "\n".join(captions)

#print("Training text length (chars):", len(text))
#print(text[:1000])

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s: str):
    return [stoi[c] for c in s]


def decode(ids):
    return "".join(itos[i] for i in ids)

# print("Size of the vocabulary:", vocab_size)
# print("Preview of the vocabulary:", chars)

# examples = ["male", "malignant", "melanoma", "malignant melanoma"]

# print("\n--- Encoding Examples ---")

# for word in examples:
#     tokens = encode(word)
    
#     # Create a visual mapping of Char -> Token
#     mapping_str = ", ".join([f"'{c}':{t}" for c, t in zip(word, tokens)])
    
#     print(f"String:  {word}")
#     print(f"Tokens:  {tokens}")
#     print(f"Mapping: {mapping_str}")
#     print("-" * 40)

data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split: str):
    src = train_data if split == "train" else val_data
    ix = torch.randint(len(src) - cfg.block_size - 1, (cfg.batch_size,))
    x = torch.stack([src[i:i + cfg.block_size] for i in ix])
    y = torch.stack([src[i + 1:i + cfg.block_size + 1] for i in ix])
    return x.to(cfg.device), y.to(cfg.device)

#print("Train tokens:", train_data.numel(), "Val tokens:", val_data.numel())

# =====================
# Configurations
# =====================

@dataclass
class ModelConfigGPU:
    vocab_size: int
    block_size: int
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.2


@dataclass
class ModelConfigCPU:
    vocab_size: int
    block_size: int
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0


@dataclass
class TrainConfigGPU:
    block_size: int = 256
    batch_size: int = 64
    max_iters: int = 2000
    eval_interval: int = 250
    eval_iters: int = 200
    lr: float = 1e-3
    weight_decay: float = 0.1
    device: str = "cuda"


@dataclass
class TrainConfigCPU:
    block_size: int = 64
    batch_size: int = 12
    max_iters: int = 2000
    eval_interval: int = 200
    eval_iters: int = 50
    lr: float = 1e-3
    weight_decay: float = 0.1
    device: str = "cpu"


if torch.cuda.is_available():
    ModelConfig = ModelConfigGPU
    TrainConfig = TrainConfigGPU
else:
    ModelConfig = ModelConfigCPU
    TrainConfig = TrainConfigCPU

cfg = TrainConfig()



# =====================
# Model Components
# =====================

class CausalSelfAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        assert c.n_embd % c.n_head == 0

        self.n_head = c.n_head
        self.head_dim = c.n_embd // c.n_head

        self.qkv = nn.Linear(c.n_embd, 3 * c.n_embd, bias=False)
        self.proj = nn.Linear(c.n_embd, c.n_embd, bias=False)
        self.dropout = nn.Dropout(c.dropout)

        mask = torch.tril(torch.ones(c.block_size, c.block_size)).view(1, 1, c.block_size, c.block_size)
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.proj(y)
        y = self.dropout(y)
        return y


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.fc = nn.Linear(c.n_embd, 4 * c.n_embd)
        self.proj = nn.Linear(4 * c.n_embd, c.n_embd)
        self.dropout = nn.Dropout(c.dropout)

    def forward(self, x):
        x = self.fc(x)
        x = F.gelu(x)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln1 = nn.LayerNorm(c.n_embd)
        self.attn = CausalSelfAttention(c)
        self.ln2 = nn.LayerNorm(c.n_embd)
        self.mlp = MLP(c)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class NanoGPT(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c

        self.tok_emb = nn.Embedding(c.vocab_size, c.n_embd)
        self.pos_emb = nn.Embedding(c.block_size, c.n_embd)

        self.drop = nn.Dropout(c.dropout)

        self.blocks = nn.ModuleList([Block(c) for _ in range(c.n_layer)])
        self.ln_f = nn.LayerNorm(c.n_embd)
        self.head = nn.Linear(c.n_embd, c.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)


    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.c.block_size
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)

        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# =====================
# Training
# =====================

@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}

    for split in ["train", "val"]:
        losses = torch.zeros(cfg.eval_iters)

        for k in range(cfg.eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()

        out[split] = losses.mean().item()

    model.train()
    return out


@torch.no_grad()
def generate(model, start, max_new_tokens=400, temperature=1.0, top_k=60):
    model.eval()
    idx = torch.tensor([encode(start)], dtype=torch.long, device=cfg.device)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -cfg.block_size:]
        logits, _ = model(idx_cond)

        logits = logits[:, -1, :] / max(temperature, 1e-6)

        if top_k is not None:
            v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        idx = torch.cat([idx, next_id], dim=1)

    return decode(idx[0].tolist())


# =====================
# MAIN EXECUTION
# =====================

if __name__ == "__main__":
    set_seed(42)
    print("Starting training on", cfg.device)

    mcfg = ModelConfig(vocab_size=vocab_size, block_size=cfg.block_size)
    model = NanoGPT(mcfg).to(cfg.device)

    print("Parameters:", sum(p.numel() for p in model.parameters()) / 1e6, "M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    pbar = tqdm(range(cfg.max_iters))

    for it in pbar:

        if it % cfg.eval_interval == 0:
            losses = estimate_loss(model)
            pbar.set_postfix(train=losses["train"], val=losses["val"])

        xb, yb = get_batch("train")
        _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    prompt = "H&E stained section showing"

    print("\n" + "="*80)
    print("RUN:", "WITH <ENDC>" if USE_ENDC else "WITHOUT <ENDC>")
    print("vocab_size:", vocab_size)
    print("="*80)

    # Generate 10 samples
    for i in range(10):
        print(f"\n--- Sample {i+1} ---")
        out = generate(model, prompt, max_new_tokens=500, temperature=0.7, top_k=10)

        if USE_ENDC:
            print("Contains <ENDC>:", "<ENDC>" in out)
            if "<ENDC>" in out:
                out = out.split("<ENDC>")[0] + "<ENDC>"

        print(out)



