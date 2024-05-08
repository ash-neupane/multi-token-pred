import math
import torch
from torch import nn
from torch.nn import functional as F

@dataclass
class ModelConfig:
    vocab_size: int = 0 # vocab size 
    block_size: int = 0 # context window size
    n_layer: int = 0 # number of layers
    n_heads: int = 0 # number of attention heads in each block
    d_model: int = 0 # model dimension (dimension of residual stream)

class VanillaTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        self.block_size = config.block_size
        self.n_layer = config.n_layer

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(self.vocab_size, self.d_model),
            wpe = nn.Embedding(self.block_size, self.d_model),
            h = nn.ModuleList(
                [TransformerBlock(config) for _ in range(self.n_layer)]
            ),
            ln_f = nn.LayerNorm(self.d_model) # final layer norm before unembed
        ))
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False) # unembed
        self.transformer.wte.weight = self.lm_head.weight # weight tying to reduce num params

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def forward(self, context, targets=None):
        device = context.device
        B, T = context.size()
        assert T <= self.block_size, f"Can't forward context w len {T}, max {self.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=device)

        # embed
        tok_emb = self.transformer.wte(context)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb

        # apply blocks
        for block in self.transformer.h:
            x = block(x)
        
        # then pre-unembed layer norm
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # at inference, only forward lm_head on the very last position 
            logits = self.lm_head(x[:, [-1], :]) # use [-1] to preserve T dimension
            loss = None

        return logits, loss
    
    @torch.no_grad
    def generate(self, context, max_new_tokens, temp=0.5, top_k=None):
        """
        """
        for _ in range(max_new_tokens):
            context_cropped = context if context.size(1) < self.block_size else context[:, -self.block_size:]
            logits, _ = self(context_cropped)
            logits = logits[:, -1, :] / temp
            # sample only from top_k if that's enabled
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            context = torch.cat((context, next_token), dim=1)
        return context

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

        self.w_attn = nn.Linear(self.d_model, 3 * config.d_model, bias=False) # 3 for Q, K, V
        self.w_proj = nn.Linear(self.d_model, self.d_model, bias=False)

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
            nn.Linear(self.dim_in_out, self.hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.dim_in_out, bias=False),
        )

    def forward(self, context):
        """
        
        context: Batch Size x Block Size
        targets: Batch Size
        """
        if context.size(1) != self.block_size:
            raise ValueError(f"Input tensor {context.size} doesn't match block size of {self.block_size}")
        return self.mlp(context)
    
