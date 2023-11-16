from torch.utils.data import Dataset
import numpy as np
import src.ctes.num_ctes as nctes
import src.ctes.str_ctes as sctes

class RegressionDataset(Dataset):
    def __init__(self,  mrw_path, transform=None, sample_size=nctes.LEN_SAMPLE):
        data = np.load(mrw_path)
        self.X = data[sctes.X]
        self.Y = data[sctes.Y]
        self.transform = transform
        self.sample_size = int(sample_size)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx:idx+1, :self.sample_size]
        y = self.Y[idx:idx+1, :]
        if self.transform:
            x = self.transform(x)
        return x, y