import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from .transformer.transformer import SelfDecoder, precompute_freqs_cis, ModelArgs


class TokenMerger(nn.Module):
    def __init__(
        self, aggregation_num: int = 4, seq_len: int = 256, hidden_dim: int = 768
    ):
        super().__init__()

        assert hidden_dim % aggregation_num == 0
        assert seq_len % aggregation_num == 0

        self.aggregation_num = aggregation_num

        self.block_norm = nn.RMSNorm(hidden_dim)
        self.token_mixer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.token_gate = nn.Linear(hidden_dim, 1)
        self.down_proj = nn.Linear(hidden_dim, hidden_dim)

        self.up_norm = nn.RMSNorm(hidden_dim)
        self.slot_embedding = nn.Parameter(torch.randn(aggregation_num, hidden_dim))
        self.up_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, tokens: torch.Tensor):
        tokens = rearrange(
            tokens, "b (num_blocks block_size) d -> b num_blocks block_size d", block_size=self.aggregation_num
        )
        tokens = self.block_norm(tokens)
        mixed_tokens = self.token_mixer(tokens)
        token_weights = torch.softmax(self.token_gate(mixed_tokens), dim=2)
        pooled_tokens = (mixed_tokens * token_weights).sum(dim=2)

        return self.down_proj(pooled_tokens)

    def upsample(self, tokens: torch.Tensor):
        tokens = self.up_proj(self.up_norm(tokens))
        tokens = tokens.unsqueeze(2) + self.slot_embedding.unsqueeze(0).unsqueeze(0)

        return rearrange(tokens, "b num_blocks block_size d -> b (num_blocks block_size) d")


class TokenAggregationModel(nn.Module):
    def __init__(
        self,
        config: ModelArgs,
        aggregation_num: int = 4,
    ):
        super().__init__()

        assert config.cls_token_num == 1

        self.aggregation_num = aggregation_num
        self.config = config

        self.embedding_layer = nn.Embedding(config.vocab_size, config.dim)
        self.token_merger = TokenMerger(
            hidden_dim=config.dim,
            aggregation_num=aggregation_num,
            seq_len=config.seq_len,
        )

        self.blocks = nn.ModuleList(
            [SelfDecoder(config=self.config) for _ in range(config.n_layer)]
        )

        self.decode_norm = nn.RMSNorm(config.dim)
        self.decode_mlp = nn.Sequential(
            nn.Linear(config.dim, config.dim * 2),
            nn.GELU(),
            nn.Linear(config.dim * 2, config.dim),
        )
        self.lm_norm = nn.RMSNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size)

        self.register_buffer(
            "freq_cis",
            precompute_freqs_cis(
                seq_len=(config.seq_len // self.aggregation_num) - 1,
                n_elem=config.dim // config.n_head,
                base=config.rope_base,
                cls_token_num=config.cls_token_num,
            ),
            persistent=False,
        )

    def forward(self, token_ids: torch.Tensor):
        _, seq_len = token_ids.shape
        if seq_len != self.config.seq_len:
            raise ValueError(
                f"Expected seq_len={self.config.seq_len}, got seq_len={seq_len}"
            )
        if seq_len % self.aggregation_num != 0:
            raise ValueError(
                f"seq_len={seq_len} must be divisible by aggregation_num={self.aggregation_num}"
            )

        target_ids = token_ids[:, self.aggregation_num :].detach()

        token_embeddings = self.embedding_layer(token_ids)
        token_merged = self.token_merger(token_embeddings)

        freq_cis = self.freq_cis.unsqueeze(0).expand(token_ids.shape[0], -1, -1, -1)
        for block in self.blocks:
            token_merged = block(token_merged, freqs_cis=freq_cis)

        token_merged = token_merged[:, :-1]
        upsampled_tokens = self.token_merger.upsample(token_merged)
        upsampled_tokens = upsampled_tokens + self.decode_mlp(
            self.decode_norm(upsampled_tokens)
        )

        logits = self.lm_norm(upsampled_tokens)
        logits = self.lm_head(logits)

        return F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size), target_ids.reshape(-1)
        )
