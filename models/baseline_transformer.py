import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer.transformer import ModelArgs, SelfDecoder, precompute_freqs_cis


class BaselineTransformerLM(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config

        self.embedding_layer = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList(
            [SelfDecoder(config=config) for _ in range(config.n_layer)]
        )
        self.lm_norm = nn.RMSNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size)

        self.register_buffer(
            "freq_cis",
            precompute_freqs_cis(
                seq_len=config.seq_len,
                n_elem=config.dim // config.n_head,
                base=config.rope_base,
                cls_token_num=0,
            ),
            persistent=False,
        )

    def forward(self, token_ids: torch.Tensor):
        _, seq_len = token_ids.shape
        if seq_len != self.config.seq_len:
            raise ValueError(
                f"Expected seq_len={self.config.seq_len}, got seq_len={seq_len}"
            )

        input_ids = token_ids[:, :-1]
        target_ids = token_ids[:, 1:]

        tokens = self.embedding_layer(input_ids)
        freqs_cis = self.freq_cis.unsqueeze(0).expand(token_ids.shape[0], -1, -1, -1)
        freqs_cis = freqs_cis[:, : input_ids.shape[1], ...]

        for block in self.blocks:
            tokens = block(tokens, freqs_cis=freqs_cis)

        logits = self.lm_head(self.lm_norm(tokens))
        return F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size), target_ids.reshape(-1)
        )
