import torch

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        if(torch.cuda.is_available()):
            return torch.FloatTensor(sample).cuda()
        else:
            return torch.FloatTensor(sample)