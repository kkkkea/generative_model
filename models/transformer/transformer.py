import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from typing import Optional
from transformers.configuration_utils import PreTrainedConfig


def find_multiple(n: int, k: int):
    if n % k == 0:
        return n
    return n + k - (n % k)


# https://github.com/pytorch-labs/gpt-fast/blob/main/model.py
def precompute_freqs_cis(
    seq_len: int, n_elem: int, base: int = 10000, cls_token_num=120
):
    freqs = 1.0 / (
        base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem)
    )
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)  # (seq_len, head_dim // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack(
        [freqs_cis.real, freqs_cis.imag], dim=-1
    )  # (cls_token_num+seq_len, head_dim // 2, 2)
    cond_cache = torch.cat(
        [torch.zeros(cls_token_num, n_elem // 2, 2), cache]
    )  # (cls_token_num+seq_len, head_dim // 2, 2)
    return cond_cache


def precompute_freqs_cis_2d(
    grid_size: int, n_elem: int, base: int = 10000, cls_token_num=120
):
    # split the dimension into half, one for x and one for y
    half_dim = n_elem // 2
    freqs = 1.0 / (
        base ** (torch.arange(0, half_dim, 2)[: (half_dim // 2)].float() / half_dim)
    )
    t = torch.arange(grid_size, device=freqs.device)
    freqs = torch.outer(t, freqs)  # (grid_size, head_dim // 2)
    freqs_grid = torch.concat(
        [
            freqs[:, None, :].expand(-1, grid_size, -1),
            freqs[None, :, :].expand(grid_size, -1, -1),
        ],
        dim=-1,
    )  # (grid_size, grid_size, head_dim // 2)
    cache_grid = torch.stack(
        [torch.cos(freqs_grid), torch.sin(freqs_grid)], dim=-1
    )  # (grid_size, grid_size, head_dim // 2, 2)
    cache = cache_grid.flatten(0, 1)
    cond_cache = torch.cat(
        [torch.zeros(cls_token_num, n_elem // 2, 2), cache]
    )  # (cls_token_num+grid_size**2, head_dim // 2, 2)
    return cond_cache


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    # x: (bs, seq_len, n_head, head_dim)
    # freqs_cis (seq_len, head_dim // 2, 2)
    xshaped = x.float().reshape(
        *x.shape[:-1], -1, 2
    )  # (bs, seq_len, n_head, head_dim//2, 2)
    freqs_cis = freqs_cis.view(
        -1, xshaped.size(1), 1, xshaped.size(3), 2
    )  # (1, seq_len, 1, head_dim//2, 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        dim=-1,
    )
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


class ModelArgs(PreTrainedConfig):
    def __init__(
        self,
        dim: int = 4096,
        n_layer: int = 32,
        n_head: int = 32,
        multiple_of: int = 256,  # make SwiGLU hidden layer size multiple of large power of 2
        ffn_dim_multiplier: Optional[float] = None,
        rope_base: float = 10000,
        norm_eps: float = 1e-5,
        initializer_range: float = 0.02,
        token_dropout_p: float = 0.1,
        attn_dropout_p: float = 0.0,
        resid_dropout_p: float = 0.1,
        ffn_dropout_p: float = 0.1,
        drop_path_rate: float = 0.0,
        num_classes: int = 1000,
        class_dropout_prob: float = 0.1,
        model_type: str = "c2i",
        vocab_size: int = 16384,
        cls_token_num: int = 1,
        block_size: int = 256,
        seq_len: int = 256,
    ):
        self.dim = dim
        self.n_layer = n_layer
        self.n_head = n_head
        self.multiple_of = multiple_of
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.rope_base = rope_base
        self.norm_eps = norm_eps
        self.initializer_range = initializer_range

        self.token_dropout_p = token_dropout_p
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout_p = resid_dropout_p
        self.ffn_dropout_p = ffn_dropout_p
        self.drop_path_rate = drop_path_rate

        self.num_classes = num_classes
        self.class_dropout_prob = class_dropout_prob
        self.model_type = model_type
        self.vocab_size = vocab_size
        self.cls_token_num = cls_token_num
        self.block_size = block_size

        self.seq_len = seq_len


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        hidden_dim = 4 * config.dim
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if config.ffn_dim_multiplier is not None:
            hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)
        hidden_dim = find_multiple(hidden_dim, config.multiple_of)

        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)
        self.ffn_dropout = nn.Dropout(config.ffn_dropout_p)

    def forward(self, x):
        return self.ffn_dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0
        self.dim = config.dim
        self.n_head = config.n_head
        self.head_dim = config.dim // config.n_head

        self.to_q = nn.Linear(config.dim, config.dim, bias=False)
        self.to_k = nn.Linear(config.dim, config.dim, bias=False)
        self.to_v = nn.Linear(config.dim, config.dim, bias=False)

        self.proj = nn.Linear(config.dim, config.dim, bias=False)

        self.attn_drop = config.attn_dropout_p
        self.proj_drop = nn.Dropout(config.resid_dropout_p)

        self.kv_cache = False
        self.k_cache = None
        self.v_cache = None

    def reset_kv_cache(self):
        self.k_cache = None
        self.v_cache = None

    def update_kv_cache(self, k: torch.Tensor, v: torch.Tensor):
        if self.k_cache is None and self.v_cache is None:
            k_cache = k
            v_cache = v
        else:
            k_cache = torch.cat([self.k_cache, k], dim=-2)
            v_cache = torch.cat([self.v_cache, v], dim=-2)

        self.k_cache = k_cache
        self.v_cache = v_cache

        return k_cache, v_cache

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor = None):

        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b n h d", h=self.n_head), (q, k, v)
        )

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if self.kv_cache:
            k, v = self.update_kv_cache(k, v)

        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True if self.training else False,
            dropout_p=self.attn_drop if self.training else 0,
        )
        output = rearrange(output, "b h n d -> b n (h d)").contiguous()
        output = self.proj_drop(self.proj(output))
        return output


class CrossAttention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0
        self.dim = config.dim
        self.n_head = config.n_head
        self.head_dim = config.dim // config.n_head

        self.to_q = nn.Linear(config.dim, config.dim, bias=False)

        self.proj = nn.Linear(config.dim, config.dim, bias=False)

        self.attn_drop = config.attn_dropout_p
        self.proj_drop = nn.Dropout(config.resid_dropout_p)

        self.kv_cache = False
        self.k_cache = None
        self.v_cache = None

    def reset_kv_cache(self):
        self.k_cache = None
        self.v_cache = None

    def update_kv_cache(self, k: torch.Tensor, v: torch.Tensor):
        if self.k_cache is None and self.v_cache is None:
            k_cache = k
            v_cache = v
        else:
            k_cache = torch.cat([self.k_cache, k], dim=-2)
            v_cache = torch.cat([self.v_cache, v], dim=-2)

        self.k_cache = k_cache
        self.v_cache = v_cache

        return k_cache, v_cache

    def forward(
        self,
        x: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        freqs_cis: torch.Tensor = None,
    ):
        q = self.to_q(x)
        q = rearrange(q, "b n (h d) -> b n h d", h=self.n_head)

        # target-aware
        q = apply_rotary_emb(q, freqs_cis[:, -q.shape[1] :, ...])

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if self.kv_cache:
            k, v = self.update_kv_cache(k, v)

        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True if self.training else False,
            dropout_p=self.attn_drop if self.training else 0,
        )
        output = rearrange(output, "b h n d -> b n (h d)").contiguous()
        output = self.proj_drop(self.proj(output))
        return output


class SelfDecoder(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.attn = Attention(config)
        self.ffn = FeedForward(config)

        self.attn_norm = nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = nn.RMSNorm(config.dim, eps=config.norm_eps)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor = None):
        h = x + self.attn(
            x=self.attn_norm(x),
            freqs_cis=freqs_cis[:, : x.shape[1], ...],
        )
        out = h + self.ffn(self.ffn_norm(h))

        return out


class CrossDecoder(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.attn = CrossAttention(config)
        self.ffn = FeedForward(config)

        self.attn_norm = nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = nn.RMSNorm(config.dim, eps=config.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        freqs_cis: torch.Tensor = None,
    ):
        h = x + self.attn(x=self.attn_norm(x), k=k, v=v, freqs_cis=freqs_cis)
        out = h + self.ffn(self.ffn_norm(h))

        return out


class Decoder_Decoder(nn.Module):
    def __init__(self, config: ModelArgs, n_layer):
        super().__init__()
        self.config = config
        self.self_dec = nn.ModuleList(
            [SelfDecoder(config) for _ in range(n_layer // 2)]
        )
        self.cross_dec = nn.ModuleList(
            [CrossDecoder(config) for _ in range(n_layer // 2)]
        )

        self.norm = nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.to_k = nn.Linear(config.dim, config.dim, bias=False)
        self.to_v = nn.Linear(config.dim, config.dim, bias=False)

        self.kv_cache = False
        self.k_cache = None
        self.v_cache = None

    def reset_kv_cache(self):
        self.k_cache = None
        self.v_cache = None

    def update_kv_cache(self, k: torch.Tensor, v: torch.Tensor, head_first=False):
        t_dim = 2 if head_first else 1

        if self.k_cache is None and self.v_cache is None:
            k_cache = k
            v_cache = v
        else:
            k_cache = torch.cat([self.k_cache, k], dim=t_dim)
            v_cache = torch.cat([self.v_cache, v], dim=t_dim)

        self.k_cache = k_cache
        self.v_cache = v_cache

        return k_cache, v_cache

    def forward(self, x: torch.Tensor, q: torch.Tensor, freqs_cis: torch.Tensor = None):
        for layer in self.self_dec:
            x = layer(x=x, freqs_cis=freqs_cis)

        x_norm = self.norm(x)
        k = self.to_k(x_norm)
        v = self.to_v(x_norm)

        k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b n h d", h=self.config.n_head), (k, v)
        )
        k = apply_rotary_emb(k, freqs_cis[:, : k.shape[1], ...])

        if self.kv_cache:
            k, v = self.update_kv_cache(k, v)

        for layer in self.cross_dec:
            q = layer(x=q, k=k, v=v, freqs_cis=freqs_cis)

        return q
