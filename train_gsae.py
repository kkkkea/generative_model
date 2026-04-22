from models.lightning.lightning_cli import (
    ReWriteRootDirCli,
    ReWriteRootSaveConfigCallback,
)
from models.lightning.lightning_data import DataModule
from models.lightning.lightning_model import GaussianAEModel


def main():
    cli = ReWriteRootDirCli(
        GaussianAEModel,
        DataModule,
        auto_configure_optimizers=False,
        save_config_callback=ReWriteRootSaveConfigCallback,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()
