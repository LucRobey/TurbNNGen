from torchsummary                    import summary
from datetime                        import datetime
from tqdm                            import tqdm
from src.data.to_tensor              import ToTensor
from src.data.mrw_dataset            import MRWDataset
from src.data.create_data_loaders    import create_data_loaders

import src.ctes.num_ctes             as nctes
import src.ctes.path_ctes            as pctes
import numpy                         as np
import torch 
import random

class Trainer():
    @classmethod
    def train(cls, n_epochs, train_loader, valid_loader, model, criterion, optimizer, device, save_path, losses_path):
        train_losses, valid_losses = [], []
        # initialize tracker for minimum validation loss
        valid_loss_min = np.Inf  # set initial "min" to infinity
            
        for epoch in range(n_epochs):
            print(f"Epoch {epoch+1}/{n_epochs} ...")
            train_loss, valid_loss = 0, 0 # monitor losses
            
            # train the model
            print("Training ...")
            model.train() # prep model for training
            for data, label in tqdm(train_loader):
                data = data.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * data.size(0) # update running training loss
            
            # validate the model
            print("Validating ...")
            model.eval()
            for data, label in tqdm(valid_loader):
                data = data.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)
                with torch.no_grad():
                    output = model(data)
                loss = criterion(output, label)
                valid_loss += loss.item() * data.size(0)
            
            # calculate average loss over an epoch
            train_loss /= len(train_loader.sampler)
            valid_loss /= len(valid_loader.sampler)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            
            print('Epoch: {} \ttraining Loss: {:.6f} \tvalidation Loss: {:.6f}'.format(epoch+1, train_loss, valid_loss))

            # save model if validation loss has decreased
            if valid_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
                torch.save(model.state_dict(), save_path)
                valid_loss_min = valid_loss

            np.savez(losses_path , train=np.array(train_losses), val=np.array(valid_losses))
        
        return train_losses, valid_losses
    
    @classmethod
    def new_training(cls, builder, criterion, optimizer_builder, optimizer_params, data_path="./data", model_name="", n_epochs=100, batch_size=6, valid_size=0.2, test_size=0.2, seed=42, verbose=False):
        if verbose: print("[TurbNNGen] New training ... ")
        
        if verbose: print("[TurbNNGen] Setting timestamp ... ", end="")
        timestamp = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
        arch_name  = builder.__name__
        hyperparams_path = f"{data_path}/hyperparams_{arch_name}_{timestamp}.npz"
        model_path = f"{data_path}/model_{arch_name}_{timestamp}.pt"
        losses_path = f"{data_path}/losses_{arch_name}_{timestamp}.npz"
        if verbose: print(f"{timestamp}")

        if verbose: print("[TurbNNGen] \t Setting seed ... ", end="")
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        if verbose: print(f"{seed}")

        if verbose: print("[TurbNNGen] \t Setting device ... ", end="")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if verbose: print(f"{str(device)}")

        if verbose: print("[TurbNNGen] \t Building model ... ", end="")
        sample_size = nctes.LEN_SAMPLE
        dropout_probs=[]
        model = builder(input_size=sample_size, dropout_probs=dropout_probs)
        model.to(device=device)
        if verbose: print(f"{arch_name}")
        if verbose: summary(model, (1, sample_size))

        if verbose: print("[TurbNNGen] \t Loading data ... ", end="")
        data_path = pctes.DATAPATH
        transform = ToTensor()
        data = MRWDataset(data_path, transform, sample_size, builder.LABELS)
        if verbose: print("OK")

        if verbose: print("[TurbNNGen] \t Spliting data ... ", end="")
        train_loader, valid_loader, _ = create_data_loaders(batch_size, valid_size, test_size, data)
        if verbose: print("OK")

        if verbose: print(f"[TurbNNGen] \t Criterion ... ")
        if verbose: print(criterion)

        optimizer_params["params"] = model.parameters()
        optimizer = optimizer_builder(**optimizer_params)
        if verbose: print(f"[TurbNNGen] \t Optimizer ... ")
        if verbose: print(optimizer)

        if verbose: print(f"[TurbNNGen] \t Saving hyperparameters ... ", end="")
        np.savez(hyperparams_path, 
            len           = len(data), 
            test_size     = test_size, 
            valid_size    = valid_size, 
            epochs        = n_epochs, 
            batch_size    = batch_size, 
            criterion     = str(criterion), 
            optimizer     = str(optimizer), 
            seed          = seed,
            dropout_probs = dropout_probs, 
            model_name    = model_name,
            arch_name     = arch_name)
        if verbose: print(f"OK")
        
        if verbose: print(f"[TurbNNGen] \t Training start ... ")
        Trainer.train(n_epochs, train_loader, valid_loader, model, criterion, optimizer, device, model_path, losses_path)
    