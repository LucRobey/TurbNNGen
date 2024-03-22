from src.nn.training.utils      import Trainer
from src.nn.archs               import CNN_ALL, CNN_ALL_VDCNNFRW_M18, Wav2Vec2ALL

import torch
import torch.nn                 as nn


if __name__ == "__main__":
    Trainer.new_training(builder=CNN_ALL_VDCNNFRW_M18, 
                        criterion=nn.MSELoss(), 
                        optimizer_builder=torch.optim.Adam, 
                        optimizer_params={"lr":1e-4, "weight_decay": 0.0}, 
                        data_path="./data/models",
                        model_name="wav2vec2", 
                        n_epochs=2,
                        batch_size=8, 
                        valid_size=0.2, 
                        test_size=0.2, 
                        seed=42, 
                        verbose=True)
    