import numpy as np
import torch
from typing import List, Tuple
import json, os

def train_test_split(data, train_ratio=0.7, test_ratio=None, return_index=False, seed=None):
    if isinstance(seed, int):
        np.random.seed(seed)
    if train_ratio is not None:
        n = int(len(data) * train_ratio)
    elif test_ratio is not None:
        n = int(len(data) * (1-test_ratio))
    else:
        raise ValueError("Neither `train_ratio` nor `test_ratio` is given")
    idxs = np.arange(len(data))
    np.random.shuffle(idxs)
    train_index, test_index = idxs[:n], idxs[n:]
    if return_index:
        return train_index, test_index
    else:
        data_ = np.array(data)
        return data_[train_index], data_[test_index]

def stratified_train_test_split(data, stratum, train_ratio=None, test_ratio=None,
                                train_stratum=None, remove_stratum=None, 
                                return_index=False, seed=None):
    train_idxs = []
    test_idxs  = []
    if isinstance(seed, int):
        np.random.seed(seed)
    if train_ratio is not None or test_ratio is not None:
        for i in np.sort(np.unique(stratum)):
            if remove_stratum is not None and i in remove_stratum: continue
            idxs = np.where(i == stratum)
            if train_ratio is not None:
                n = int(len(idxs) * train_ratio)
            else:
                n = int(len(idxs) * (1 - test_ratio))
            train_idxs.append(idxs[:n])
            test_idxs.append(idxs[n:])
    elif train_stratum is not None:
        stratum = np.array(stratum)
        for i in np.unique(stratum):
            if remove_stratum is not None and i in remove_stratum: continue
            if i in train_stratum:
                train_idxs.append(np.where(i == stratum)[0])
            else:
                test_idxs.append(np.where(i == stratum)[0])
    else:
        raise ValueError("Neither `train_ratio/test_ratio` nor `train_stratum` is given")
    train_idxs = np.hstack(train_idxs)
    test_idxs  = np.hstack(test_idxs)
    if return_index:
        return train_idxs, test_idxs
    else:
        data_ = np.array(data)
        return data_[train_idxs], data_[test_idxs]

def train_test_split_by_smiles(data, n_test=1, seed=None, min_train_ratio=0.7, test_smiles=None):
    if seed:
        np.random.seed(seed)
    data_  = np.array(data)
    n_data = len(data_)
    if isinstance(seed, int):
        np.random.seed(seed)
    smiles = np.vstack([d['smiles'] for d in data_])
    unique = sorted(set(smiles.reshape(-1)))
    if test_smiles is not None:
        test_mask = np.zeros_like(smiles, dtype=bool)
        for s in test_smiles:
            test_mask = test_mask | (smiles == s)
        test_mask = np.sum(test_mask, axis=1) != 0
    else:
        while True:
            np.random.shuffle(unique)
            test_smiles = unique[:n_test]
            test_mask  = np.zeros_like(smiles, dtype=bool)
            for s in test_smiles:
                test_mask = test_mask | (smiles == s)
            test_mask = np.sum(test_mask, axis=1) != 0
            if np.sum(~test_mask) > n_data*min_train_ratio:
                break
    return data_[~test_mask], data_[test_mask], test_smiles

class CrossValidation:
    def __init__(self, n_fold:int, data=None, n_data=None, stratum=None, return_index=False, seed:int=None):
        if data is None and n_data is None:
            raise ValueError('Either `n_data` or `data` should be given')
        if data is None:
            return_index = True
        if isinstance(seed, int):
            np.random.seed(seed)

        self.return_index = return_index
        if data is not None:
            n_data = len(data)
            self._data = np.array(data)

        index = np.arange(n_data)
        if stratum is None:
            k = np.min([n_fold, n_data])
            if k < n_fold:
                print(f"Notice: Reduced number of folds ({n_fold} -> {k})")
            np.random.shuffle(index)
            self.train_index, self.valid_index = self._split_(index, k)
        else:
            if len(stratum) != n_data:
                raise ValueError(f"Dimension mismatch between `data` ({n_data}) and `stratum` ({len(stratum)})")
            stratum = np.array(stratum)
            cs = np.sort(np.unique(stratum))
            ks = np.array([np.sum(c == stratum) for c in cs])
            k  = np.min([n_fold, np.max(ks)])
            if k < n_fold:
                print(f"Notice: Reduced number of folds ({n_fold} -> {k})")
            train_index = [[] for _ in range(k)]
            valid_index = [[] for _ in range(k)]
            if k > np.min(ks):
                print(f"Warning: Number of folds is larger than number of data (got: {k} / min: {np.min(ks)}).\nMore than one fold contains full data of class: {cs[ks < n_fold]}")
            for c in cs:
                idx = index[c == stratum]
                np.random.shuffle(idx)
                tidxs, vidxs = self._split_(idx, k)
                for i, (tidx, vidx) in enumerate(zip(tidxs, vidxs)):
                    train_index[i].append(tidx)
                    valid_index[i].append(vidx)
            self.train_index = [np.hstack(idx) for idx in train_index]
            self.valid_index = [np.hstack(idx) for idx in valid_index]

    def __getitem__(self, i:int) -> Tuple[List[int], List[int]]:
        if self.return_index:
            return self.train_index[i], self.valid_index[i]
        else:
            return self._data[self.train_index[i]], self._data[self.valid_index[i]]

    def __len__(self):
        return len(self.train_index)

    def _split_(self, index, k):
        n = len(index)
        c = int(n/k) + bool(n%k)
        l = n - k * (c - bool(n%k))
        train_index = []
        valid_index = []
        i1 = 0
        i2 = c
        for i in range(k):
            train_index.append(np.hstack([index[:i1], index[i2:]]))
            valid_index.append(index[i1:i2])

            i1 += c
            if i == (l-1): c -= 1
            i2 += c
        return train_index, valid_index

class DataScaler:
    def __init__(self, device='cpu'):
        self._avg = torch.tensor([0]).float()
        self._div = torch.tensor([1]).float()
        self._device = device

    def load(self, path, fn='scale.json'):
        with open(os.path.join(path, fn)) as f:
            vals = json.load(f)
        self._avg = torch.tensor(vals['avg'], device=self._device).float()
        self._div = torch.tensor(vals['div'], device=self._device).float()

    def save(self, path, fn='scale.json'):
        with open(os.path.join(path, fn), 'w') as f:
            json.dump({
                'avg':self._avg.cpu().numpy().tolist(),
                'div':self._div.cpu().numpy().tolist(),
            }, f, indent=4)

    def train(self, data, collate_fn, index=None, cutoff=0.01):
        N = len(data) if index is None else len(index)
        i_cut = int(N * np.clip(cutoff, 0, 0.3))
        _, target, _ = collate_fn(data)
        target = target.cpu()
        if index is not None:
            target = target[index]
        self._avg = []
        self._div = []
        avgs = target.mean(dim=0)
        for avg_, dat_ in zip(avgs, target.T):
            i   = np.abs(dat_ - avg_).argsort()[-i_cut - 1]
            div = np.abs(dat_[i] - avg_).unsqueeze(0)
            avg = dat_[(dat_ <= avg_ + div) & (dat_ >= avg_ - div)].mean().unsqueeze(0)
            self._avg.append(avg)
            self._div.append(div)
        self._avg = torch.concat(self._avg).float().to(self._device)
        self._div = torch.concat(self._div).float().to(self._device)

    def scale(self, targets):
        targets.to(self._device)
        return (targets - self._avg) / self._div

    def restore(self, targets):
        targets.to(self._device)
        return targets * self._div + self._avg
