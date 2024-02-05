import numpy as np
import torch


def get_MSEs(targets, preds, output_losses, labels, device, criterion):
    test_losses = [] 
    for target, pred in zip(targets, preds):
        target = torch.FloatTensor(target).to(device=device)
        pred = torch.FloatTensor(pred).to(device=device)
        test_losses.append(criterion(target, pred).item())
    test_losses = np.array(test_losses)
    idx_sort    = np.flip(np.argsort(test_losses))
    test_losses = test_losses[idx_sort]
    targets     = targets[idx_sort]
    preds       = preds[idx_sort]

    total_test_loss = np.mean(test_losses)
    
    output_labels = {i: label for i, label in enumerate(labels)}
    mean_output_losses = {output_labels[i]: np.mean(output_losses[i]) for i in range(len(output_losses))}
    return total_test_loss, mean_output_losses
