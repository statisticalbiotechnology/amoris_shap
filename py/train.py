import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from model import AgeRegressor
from data_module import AgeDataModule
from utils import load_data

def main():
    wandb_logger = WandbLogger(project="amoris-age-prediction", entity="kall")
    df_norm, scaler, ids, status, ages = load_data("../data/NH3.csv")
    data_module = AgeDataModule(df_norm.drop("age", axis=1), ages)
    model = AgeRegressor(input_dim=df_norm.shape[1] - 1)
    trainer = pl.Trainer(max_epochs=50, logger=wandb_logger)
    trainer.fit(model, data_module)

    # Save model
    trainer.save_checkpoint("amoris_model.ckpt")

if __name__ == "__main__":
    main()
