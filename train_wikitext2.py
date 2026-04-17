import lightning.pytorch as pl

from dataset.wikitext2 import WikiText2Dataset
from models.lightning.lightning_data import DataModule
from models.lightning.lightning_model import LitTAM
from models.transformer.transformer import ModelArgs


def main():
    seq_len = 256
    aggregation_num = 4

    train_dataset = WikiText2Dataset(split="train", seq_len=seq_len)
    val_dataset = WikiText2Dataset(split="validation", seq_len=seq_len)

    data_module = DataModule(
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        train_batch_size=32,
        train_num_workers=4,
        train_prefetch_factor=4,
        eval_batch_size=32,
        eval_num_workers=4,
    )

    config = ModelArgs(
        dim=256,
        n_layer=6,
        n_head=8,
        vocab_size=train_dataset.tokenizer.vocab_size,
        seq_len=seq_len,
        cls_token_num=1,
    )

    model = LitTAM(
        model_config=config,
        aggregation_num=aggregation_num,
        lr=3e-4,
    )

    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="auto",
        devices="auto",
        precision="16-mixed",
        log_every_n_steps=20,
    )
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
