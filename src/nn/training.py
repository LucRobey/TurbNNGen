from tqdm import tqdm
import torch 
import numpy as np

def training(n_epochs, train_loader, valid_loader, model, criterion, optimizer, device, save_path):

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
            label = label.to(device=device, dtype=torch.long)
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
      
    return train_losses, valid_losses  