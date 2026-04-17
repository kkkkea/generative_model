import torch

from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class WikiText2Dataset(Dataset):
    def __init__(
        self,
        split: str,
        seq_len: int,
        tokenizer_name: str = "gpt2",
        data_path: str | None = None,
    ):
        self.seq_len = seq_len
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if data_path is None:
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
            text = "\n\n".join(
                sample["text"] for sample in dataset if sample["text"].strip()
            )
        else:
            dataset = load_dataset("parquet", data_files={split: data_path}, split=split)
            text = "\n\n".join(
                sample["text"] for sample in dataset if sample["text"].strip()
            )

        token_ids = self.tokenizer(text, return_attention_mask=False)["input_ids"]
        token_ids = torch.tensor(token_ids, dtype=torch.long)

        usable_len = (token_ids.numel() // seq_len) * seq_len
        self.samples = token_ids[:usable_len].view(-1, seq_len)

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        token_ids = self.samples[idx]
        return token_ids, None, {}
