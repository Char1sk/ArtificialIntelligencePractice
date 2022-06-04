import numpy as np


def calMIOU(pred, label, ignore_unknown=False):
    nc = pred.shape[1]
    _, index = pred.max(dim=1)
    index, label = index.cpu().numpy(), label.cpu().numpy()
    miou = 0.0
    for i, l in zip(index, label):
        hist = np.zeros((nc-1, nc-1), dtype=int)
        for ri, rl in zip(i, l):
            hist += np.bincount(nc*rl.astype(int)+ri, minlength=nc**2).reshape(nc, nc)[:-1,:-1]
        if ignore_unknown:
            hist = hist[:-1,:-1]
        miou += (np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + 1) ).mean()
    return miou

def calPA(pred, label):
    nc = pred.shape[1]
    _, index = pred.max(dim=1)
    index, label = index.cpu().numpy(), label.cpu().numpy()
    pa = 0.0
    for i, l in zip(index, label):
        hist = np.zeros((nc, nc), dtype=int)
        for ri, rl in zip(i, l):
            hist += np.bincount(nc*rl.astype(int)+ri, minlength=nc**2).reshape(nc, nc)
        pa += np.diag(hist).sum() / np.sum(hist)
    return pa
