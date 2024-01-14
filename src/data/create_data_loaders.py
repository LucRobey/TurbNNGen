import torch
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


def create_data_loaders(batch_size, valid_size, test_size, data):
    size = len(data)
    index = list(range(size))
    np.random.shuffle(index)

    test_split = int(np.floor(test_size * size))
    train_valid_index, test_index = index[test_split:], index[:test_split]

    train_valid_size = len(train_valid_index)
    
    valid_split = int(np.floor(valid_size * train_valid_size))
    train_index, valid_index = train_valid_index[valid_split:], train_valid_index[:valid_split]
    
    train_sampler = SubsetRandomSampler(train_index)
    valid_sampler = SubsetRandomSampler(valid_index)
    test_sampler = SubsetRandomSampler(test_index)

    train_loader = torch.utils.data.DataLoader(data, batch_size = batch_size, sampler = train_sampler)
    valid_loader = torch.utils.data.DataLoader(data, batch_size = batch_size, sampler = valid_sampler)
    test_loader = torch.utils.data.DataLoader(data, batch_size = batch_size, sampler = test_sampler)
    
    return train_loader, valid_loader, test_loader