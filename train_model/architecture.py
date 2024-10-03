import torch
import torch.nn as nn 
import torch.nn.functional as F 
from typing import Optional



device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.set_default_tensor_type(torch.BFloat16Tensor)





class ModelArgs:
    d_model: int = 4096 
    n_encoders: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = 8
    vocab_size: int = 128256
    multiple_of: int = 256 
    ffn_dim: Optional[float] = 14336
    norm_eps: float = 1e-05
    
    max_batch_size: int = 4
    max_seq_length: int = 128

    n_kv_heads_rep: int = n_heads // n_kv_heads
    
    head_dim: int = d_model // n_heads

    rope_theta: float = 500000.0 

    device: str = None






### Rotary Positional Embedding


def percompute_freqs_cis(d_model, end, theta = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, d_model, 2)[: (d_model // 2)].float() / d_model))
    t = torch.arange(end, device=freqs.device, dtype = torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def reshape_for_broadcast(freqs_cis, x):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)















### RMS Normalization


class RMSNorm(nn.Module):
    def __init__(self, d_model, norm_eps):
        super().__init__()
        self.norm_esp = norm_eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim = True) + self.norm_esp)
    
    def forward(self, x):
        out = self._norm(x.float()).type_as(x)
        return out * self.weight








### Feed Forward Neural Network


class FFNN(nn.Module):
    def __init__(self, args : ModelArgs):
        super().__init__()
        self.w1 = nn.Linear(args.d_model, args.ffnn_dim, bias=False)
        self.w3 = nn.Linear(args.d_model, args.ffnn_dim, bias=False)
        self.w2 = nn.Linear(args.ffnn_dim,  args.d_model, bias=False)

    def forward(self, x):
        x = self.w2(F.silu(self.w1(x)) * self.w3(x))
        return x









### Grouped Query Attention


class Grouped_Query_Attention(nn.Module):
    def __init__(self, args : ModelArgs):
        super().__init__()

        self.wq = nn.Linear(args.d_model, args.n_heads*args.head_dim, bias=False)
        self.wk = nn.Linear(args.d_model, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.d_model, args.n_kv_heads * args.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads*args.head_dim, args.d_model, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_length, args.n_kv_heads, args.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_length, args.n_kv_heads, args.head_dim))

        self.n_heads = args.n_heads
        self.head_dim = args.head_dim
        self.n_kv_heads = args.n_kv_heads


    def forward(self, x, start_pos, freqs_cis, mask):
        bsz, seqlen, _ = x.shape
        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)
        
        queries = queries.view(bsz, seqlen, self.n_heads, self.head_dim)
        keys = keys.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        values = values.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        queries, keys = apply_rotary_emb(queries, keys, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(queries.device)
        self.cache_v = self.cache_v.to(queries.device)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = keys
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = values

        keys = self.cache_k[:bsz, :start_pos + seqlen]
        values = self.cache_v[:bsz, :start_pos + seqlen]

        keys = torch.repeat_interleave(keys, dim=2, repeats=self.n_kv_heads_rep)
        values = torch.repeat_interleave(values, dim=2, repeats=self.n_kv_heads_rep)

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        out = F.scaled_dot_product_attention(queries, keys, values, attn_mask=mask)

        out = out.transpose(1,2).contiguous().view(bsz, seqlen, -1)

        return self.wo(out)











### Encoder Block


class Transformer_Encoder(nn.Module):
    def __init__(self, args:ModelArgs):
        super().__init__()
        self.attention = Grouped_Query_Attention()
        self.feed_forward = FFNN()
        self.attention_norm = RMSNorm(args.d_model, args.norm_eps)
        self.ffnn_norm = RMSNorm(args.d_model, args.norm_eps)

    def forward(self, x, start_pos, freqs_cis, mask):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffnn_norm(h))
        return out 










### Transformer


class Transformer(nn.Module):
    def __init__(self, args : ModelArgs):
        super().__init__()

        self.token_emb = nn.Embedding(args.vocab_size, args.d_model)

        self.layers = torch.nn.ModuleList()

        for _ in range(args.n_encoders):
            self.layers.append(Transformer_Encoder())

        self.norm = RMSNorm(args.d_model, args.norm_eps)

        self.linear = nn.Linear(args.d_model, args.vocab_size, bias=False)

        self.freqs_cis = percompute_freqs_cis(args.head_dim, args.max_seq_length*2, args.rope_theta)

    @torch.inference_mode()
    def forward(self, tokens, start_pos):
        _bsz, seqlen = tokens.shape
        h = self.token_emb(tokens)
        self.freqs_cis = self.freqs_cis.to(tokens.device)
        
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None 

        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1).to(tokens.device)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        
        h = self.norm(h)

        out = self.linear(h).float()

        return out 