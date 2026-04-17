import lightning.pytorch as pl
import torch

from models.tam import TokenAggregationModel
from models.transformer.transformer import ModelArgs


class LitTAM(pl.LightningModule):
    def __init__(
        self,
        model_config: ModelArgs,
        aggregation_num: int = 4,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model_config"])

        self.model = TokenAggregationModel(
            config=model_config,
            aggregation_num=aggregation_num,
        )
        self.lr = lr
        self.weight_decay = weight_decay

    def training_step(self, batch, batch_idx):
        x, _, _ = batch
        loss = self.model(x)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _, _ = batch
        loss = self.model(x)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val_ppl", torch.exp(loss), prog_bar=True, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.95),
            weight_decay=self.weight_decay,
        )
