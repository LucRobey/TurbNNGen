from src.nn.training.utils      import Trainer
from src.nn.archs               import CNN_ALL

import torch
import torch.nn                 as nn


if __name__ == "__main__":
    Trainer.new_training(builder=CNN_ALL, 
                        criterion=nn.MSELoss(), 
                        optimizer_builder=torch.optim.Adam, 
                        optimizer_params={"lr":1e-3, "weight_decay":0.0}, 
                        data_path="./data",
                        model_name="Top-Down v1", 
                        n_epochs=2, 
                        batch_size=6, 
                        valid_size=0.2, 
                        test_size=0.2, 
                        seed=42, 
                        verbose=True)
    