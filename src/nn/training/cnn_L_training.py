from src.nn.training.utils      import Trainer
from src.nn.archs               import CNN_L

import torch
import torch.nn                 as nn


if __name__ == "__main__":
    Trainer.new_training(builder=CNN_L, 
                        criterion=nn.MSELoss(), 
                        optimizer_builder=torch.optim.Adam, 
                        optimizer_params={"lr":1e-3, "weight_decay":0.0}, 
                        data_path="./data",
                        model_name="...", 
                        n_epochs=100, 
                        batch_size=6, 
                        valid_size=0.2, 
                        test_size=0.2, 
                        seed=42, 
                        verbose=False)
    