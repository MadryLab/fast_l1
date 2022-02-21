from fast_l1 import regressor
import torch as ch
from torch.utils.data import DataLoader, Dataset

# Make dataset
class BigDataset(Dataset):
    def __init__(self):
        self.w = ch.rand(1000, 10)

    def __getitem__(self, i):
        ch.manual_seed(i)
        x = ch.randn(1000)
        y = x @ self.w
        return x.cuda(), y.cuda(), i
    
    def __len__(self):
        return 10000

weight = ch.randn(1000, 10).cuda()
bias = ch.randn(10).cuda()
ds = BigDataset()
loader = DataLoader(ds, batch_size=100, shuffle=True)
regressor.train_saga(weight, bias, loader, loader, lr=1e-4, 
    start_lams=ch.ones(10).cuda(), lam_decay=0.5, end_lams=ch.ones(10).cuda())