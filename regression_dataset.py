from torch.utils.data import Dataset
import numpy as np
import ctes

class RegressionDataset(Dataset):
    def __init__(self,  mrw_path, transform=None):
        data = np.load(mrw_path)
        self.X = data[ctes.X]
        self.Y = data[ctes.S]
        self.transform = transform

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx, :]
        y = self.Y[idx, :]
        if self.transform:
            sample = self.transform(sample)
        return x, y