import lightning.pytorch as pl
import os

from dataset.wikitext2 import WikiText2Dataset
from lightning.pytorch.loggers import WandbLogger
from models.lightning.lightning_data import DataModule
from models.lightning.lightning_model import LitTAM
from models.transformer.transformer import ModelArgs


def main():
    seq_len = 256
    aggregation_num = 4
    train_data_path = os.environ.get("WIKITEXT2_TRAIN")
    val_data_path = os.environ.get("WIKITEXT2_VALIDATION")
    wandb_project = os.environ.get("WANDB_PROJECT", "tma-ar-test")
    wandb_name = os.environ.get("WANDB_NAME", "tam-wikitext2-bs64")
    wandb_entity = os.environ.get("WANDB_ENTITY")

    train_dataset = WikiText2Dataset(
        split="train", seq_len=seq_len, data_path=train_data_path
    )
    val_dataset = WikiText2Dataset(
        split="validation", seq_len=seq_len, data_path=val_data_path
    )

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

    logger = None
    if wandb_project:
        logger = WandbLogger(
            project=wandb_project,
            name=wandb_name,
            entity=wandb_entity,
            log_model=False,
        )
        logger.log_hyperparams(
            {
                "seq_len": seq_len,
                "aggregation_num": aggregation_num,
                "dim": config.dim,
                "n_layer": config.n_layer,
                "n_head": config.n_head,
                "vocab_size": config.vocab_size,
                "lr": model.lr,
                "weight_decay": model.weight_decay,
                "train_batch_size": data_module.train_batch_size,
                "eval_batch_size": data_module.eval_batch_size,
            }
        )

    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="auto",
        devices="auto",
        precision="16-mixed",
        log_every_n_steps=20,
        logger=logger,
    )
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
