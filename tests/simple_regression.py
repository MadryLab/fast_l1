from fast_l1 import regressor
import torch as ch
from torch.utils.data import DataLoader, Dataset


class BigDataset(Dataset):
    def __init__(self, w, train=False):
        self.w = w
        self.train = train

    def __getitem__(self, i):
        ch.manual_seed(i)
        if self.train:
            ch.manual_seed(len(self) + i)
        x = ch.randn(10000, device=ch.device('cuda:0'))
        eps = ch.randn(10, device=ch.device('cuda:0'))
        y = x @ self.w + eps
        return x, y, i

    def __len__(self):
        return 5000


w_star = ch.randn(10000, 10).cuda()
w_star[:9900] = 0.

ds = BigDataset(w_star, train=True)
val_ds = BigDataset(w_star, train=False)

loader = DataLoader(ds, batch_size=1000, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1000, shuffle=True)

max_lambda = regressor.calc_max_lambda(loader)
weight = ch.zeros(10000, 10).cuda()
bias = ch.zeros(10).cuda()
best_lambdas = regressor.train_saga(weight,
                                    bias,
                                    loader,
                                    val_loader,
                                    lr=1e-2,
                                    start_lams=max_lambda,
                                    lam_decay=0.5,
                                    num_lambdas=20,
                                    early_stop_eps=1e-10,
                                    early_stop_freq=10)

print((weight * w_star).sum() / weight.norm() / w_star.norm())
