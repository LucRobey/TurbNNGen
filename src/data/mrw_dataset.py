from torch.utils.data import Dataset
from src.data.to_tensor import ToTensor
import numpy as np
import src.ctes.num_ctes as nctes
import src.ctes.str_ctes as sctes


class MRWDataset(Dataset):
    IDX_LABELS = {
        sctes.C1     : 0,
        sctes.C2     : 1,
        sctes.L      : 2,
        sctes.EPSILON: 3
    }
    def __init__(self,
                 mrw_path, 
                 transform   = ToTensor(), 
                 sample_size = nctes.LEN_SAMPLE, 
                 labels      = [sctes.C1, sctes.C2, sctes.L, sctes.EPSILON]):
        data = np.load(mrw_path)
        self.X = data[sctes.X]
        idx_labels = [self.IDX_LABELS[label] for label in labels]
        self.Y = data[sctes.Y][:, idx_labels]
        self.transform = transform
        self.sample_size = int(sample_size)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx:idx+1, :self.sample_size]
        y = self.Y[idx, :]
        if self.transform:
            x = self.transform(x)
        return x, y