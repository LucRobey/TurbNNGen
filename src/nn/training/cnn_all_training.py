from src.nn.training.utils      import Trainer
from src.nn.archs               import CNN_ALL, CNN_ALL_VDCNNFRW_M18

import torch
import torch.nn                 as nn


if __name__ == "__main__":
    Trainer.new_training(builder=CNN_ALL_VDCNNFRW_M18, 
                        criterion=nn.MSELoss(), 
                        optimizer_builder=torch.optim.Adam, 
                        optimizer_params={"lr":1e-4, "weight_decay": 0.0}, 
                        data_path="./data",
                        model_name="Top-Down v2", 
                        n_epochs=2, # 100 - 400
                        batch_size=6, 
                        valid_size=0.2, 
                        test_size=0.2, 
                        seed=42, 
                        verbose=True)
    