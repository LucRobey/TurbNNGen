from tqdm import tqdm
import torch

def get_all_predictions(model, loader, device):
    preds = torch.tensor([], dtype=torch.long)
    targets = torch.tensor([], dtype=torch.long)
    for data, label in tqdm(loader):
        data = data.to(device=device, dtype=torch.float32)
        label = label.to(device=device, dtype=torch.float32)
        with torch.no_grad():
            output = model(data)
        targets = torch.cat((targets, label.cpu()), dim = 0)
        preds = torch.cat((preds, output.cpu()), dim = 0)
    return targets.numpy(), preds.numpy()