import math
import torch
from torch import nn
from torch.nn import functional as F

@dataclass
class ModelConfig:
    vocab_size: int = 0 # vocab size 
    n_embed: int = 0 # embedding dimension
    block_size: int = 0 # context window size
    n_layer: int = 0 # number of layers
    n_heads: int = 0 # number of attention heads in each block
    d_model: int = 0 # model dimension (dimension of residual stream)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.ln1 = nn.LayerNorm(self.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(self.d_model)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class CausalSelfAttention(nn.Module):
    """
    """
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        assert self.d_model % self.n_heads == 0, "d_model has to be a multiple of n_heads"
        self.attn_dim = self.d_model // self.n_heads

        self.w_attn = nn.Linear(self.d_model, 3 * config.d_model) # 3 for Q, K, V
        self.w_proj = nn.Linear(self.d_model, self.d_model)

    def forward(self, context):
        """
        """
        if context.size(1) != self.block_size:
            raise ValueError(f"Input tensor {context.size} doesn't match block size of {self.block_size}") 
        B, T, C = context.size() # batch size, sequence length, embedding dimensionality (n_embd)
        q, k, v = self.w_attn(context).split(self.d_model, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        y = self.w_proj(y)
        return y

class MLP(nn.Module):
    """
    """
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.dim_in_out = config.d_model
        self.hidden_dim = config.d_model * 4
        self.mlp = nn.Sequential(
            nn.Linear(self.dim_in_out, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.dim_in_out),
        )

    def forward(self, context):
        """
        
        context: Batch Size x Block Size
        targets: Batch Size
        """
        if context.size(1) != self.block_size:
            raise ValueError(f"Input tensor {context.size} doesn't match block size of {self.block_size}")
        return self.mlp(context)
    
