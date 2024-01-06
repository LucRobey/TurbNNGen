from tqdm import tqdm
import torch
import numpy as np

def get_all_predictions(model, loader, device, criterion):
    preds = torch.tensor([], dtype=torch.float32)
    targets = torch.tensor([], dtype=torch.float32)
    output_losses = {0: [], 1: [], 2: [], 3: []} 

    model.eval()  # evaluation mode
    with torch.no_grad():
        for data, label in tqdm(loader):
            data = data.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            output = model(data)
            
            targets = torch.cat((targets, label.cpu()), dim=0)
            preds = torch.cat((preds, output.cpu()), dim=0)

            # Calculate individual losses for each output
            for i in range(output.shape[1]):
                output_losses[i].append(criterion(output[:, i], label[:, i]).item())

    # Convert the output losses to numpy arrays
    for i in range(len(output_losses)):
        output_losses[i] = np.array(output_losses[i])

    return targets.numpy(), preds.numpy(), output_losses