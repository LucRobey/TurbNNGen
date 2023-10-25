from torch.utils.data import Dataset
import numpy as np
import num_ctes as nctes
import ctes

class RegressionDataset(Dataset):
    def __init__(self,  mrw_path, transform=None, sample_size=nctes.N):
        data = np.load(mrw_path)
        self.X = data[ctes.X]
        self.Y = data[ctes.S]
        self.transform = transform
        self.sample_size = int(sample_size)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx, :self.sample_size]
        y = self.Y[idx, :]
        if self.transform:
            sample = self.transform(sample)
        return x, y