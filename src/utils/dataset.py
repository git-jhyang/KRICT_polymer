import numpy as np
import pandas as pd
from rdkit import Chem
from torch.utils.data import Dataset
import tqdm, torch, pickle, os, json
from .features import RDKitMoleculeDescriptor, RDKitAtomicGraphDescriptor
from typing import List, Union
from abc import abstractmethod

class BaseDataset(Dataset):
    def __init__(self, norm=True, blacklist=[], **kwargs):
        super(BaseDataset, self).__init__()
        self._norm = norm        
        self._blacklist = blacklist
        if isinstance(blacklist, str) and blacklist.endswith('.json'):
            if os.path.isfile(blacklist):
                self._blacklist = json.load(open(blacklist))
            else:
                self._blacklist = []

        if isinstance(self._blacklist, dict):
            self._RDMD = RDKitMoleculeDescriptor(norm=norm, blacklist=self._blacklist['mol_feat'], **kwargs)
            self._RDGD = RDKitAtomicGraphDescriptor(norm=norm, blacklist=self._blacklist['atom_feat']+self._blacklist['bond_feat'], **kwargs)
        else:
            self._RDMD = RDKitMoleculeDescriptor(norm=norm, blacklist=self._blacklist, **kwargs)
            self._RDGD = RDKitAtomicGraphDescriptor(norm=norm, blacklist=self._blacklist, **kwargs)
        self._data = []
        self._atom_feat_name = self._RDGD.atom_descriptors
        self._bond_feat_name = self._RDGD.bond_descriptors
        self._mol_feat_name = self._RDMD.descriptors
        self._target_desc = []
        
    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        return self._data[i]

    @property
    def data(self):
        return self._data

    @property
    def atom_feat_name(self):
        return self._atom_feat_name

    @property
    def bond_feat_name(self):
        return self._bond_feat_name

    @property
    def mol_feat_name(self):
        return self._mol_feat_name

    @property
    def num_atom_feat(self):
        return len(self._atom_feat_name)

    @property
    def num_bond_feat(self):
        return len(self._bond_feat_name)

    @property
    def num_mol_feat(self):
        return len(self._mol_feat_name)

    @property
    def target_description(self):
        return self._target_desc

    @property
    def num_target(self):
        return len(self._target_desc)

    def save(self, path, overwrite=False):
        if os.path.isfile(path) and not overwrite:
            raise FileExistsError(path)
        self._data = to_numpy(self._data)
        with open(path, 'wb') as f:
            if hasattr(self, '_RDMD'): del self._RDMD
            if hasattr(self, '_RDGD'): del self._RDGD
            pickle.dump(self.__dict__, f)

    def load(self, path, device='cpu', verbose=True):
        with open(path, 'rb') as f:
            d = pickle.load(f)
        if '_blacklist' in d.keys():
            blacklist = d['_blacklist']
        else:
            blacklist = []
        cls = BaseDataset(norm=d['_norm'], blacklist=blacklist)
        for k, v in d.items():
            if verbose:
                if hasattr(self, k) and getattr(self, k) is not None:
                    print(f'  Overwriting attribute : {k[1:]}')
                elif not hasattr(self, k):
                    print(f'  Setting new attribute : {k[1:]}')
            setattr(cls, k, v)
        
        cls._data = to_tensor(cls._data, device=device)
        self.__dict__.update(cls.__dict__)

    def _generate(self, smiles, target=None, ids=None):
        data = []
        if target is None: target = np.zeros(len(smiles))
        if ids is None: ids = np.arange(len(smiles))
        pbar = None
        if len(smiles) > 5000:
            pbar = tqdm.tqdm(desc='Generate data', total=len(smiles))
        for i, s, t in zip(ids, smiles, target):
            m = Chem.MolFromSmiles(s)
            mol_feat = self._RDMD.get(m)
            atom_feat, bond_feat, bond_idx = self._RDGD.get(m)
            data.append(
                PolyData(
                    id = str(i),
                    smiles = s,
                    atom_feat = torch.tensor(atom_feat).float(),
                    bond_feat = torch.tensor(bond_feat).float(),
                    bond_idx = torch.tensor(bond_idx).long(),
                    mol_feat = torch.tensor(mol_feat).float(),
                    graph_idx = torch.zeros((atom_feat.shape[0], 1)).long(),
                    target = torch.tensor(t).view(1,-1).float(),
                )
            )
            if pbar is not None:
                pbar.update(1)
        self._data = np.array(data)
    
    def _check_cache(self, path):
        root = '/'.join(os.path.split(path)[:-1])
        fn   = os.path.split(path)[-1]
        self._cache_fn = os.path.join(root, f'cache_{fn}_{self._tag}.pkl')
        if os.path.isfile(self._cache_fn):
            return True
        else:
            return False

    @property
    def empty_data(self):
        return PolyData(
                    id = 'dummy',
                    smiles = 'dummy',
                    atom_feat = torch.zeros((1,self.num_atom_feat)).float(),
                    bond_feat = torch.zeros((0,self.num_bond_feat)).float(),
                    bond_idx = torch.zeros((2,0)).long(),
                    mol_feat = torch.zeros((1,self.num_mol_feat)).float(),
                    graph_idx = torch.zeros((1, 1)).long(),
                    weight = torch.zeros((1,1)).float(),
                    target = torch.zeros((1,1)).float(),
               )

    @abstractmethod
    def generate(self):
        pass

    def to(self, device):
        self._data = to_tensor(self._data, device)

class QM9Dataset(BaseDataset):
    def __init__(self, norm=True, **kwargs):
        super(QM9Dataset, self).__init__(norm=norm, **kwargs)

    def generate(self, 
                 path:str,
                 col_id:str='id',
                 col_smiles:Union[List[str], str] = ['smiles'],
                 col_target:Union[List[str], str] = [
                     'mu','alpha','homo','lumo','gap','R^2',
                     'zpve','Cv','U0_pa','U_pa','H_pa','G_pa'],
                 overwrite=False,
                 **kwargs):

        col_target = col_target if isinstance(col_target, List) else [col_target]
        self._target_desc = col_target
        df = pd.read_csv(path)
        df = df[np.sum(df[col_target].isna().values, axis=1) == 0]
        ids = df[col_id].values
        smiles = df[col_smiles].values.reshape(-1)
        target = df[col_target].values

        mask = [c in np.hstack([col_id, col_smiles, col_target]).reshape(-1) for c in df.columns]
#        mask += [self.num_atom_feat, self.num_bond_feat, self.num_mol_feat, self.num_target]
        self._tag = 'n' if self._norm else 'r'
        n_black = 0
        if isinstance(self._blacklist, dict):
            for v in self._blacklist.values():
                n_black += len(v)
        else:
            n_black = len(self._blacklist)
        if n_black != 0:
            self._tag += f'b{str(hex(n_black))}'
        self._tag += str(hex(int(''.join(np.array(mask).astype(int).astype(str).tolist()), 2))) 
        self._tag += '_' + str(hex(len(smiles)))

        if self._check_cache(path) and not overwrite:
            self.load(self._cache_fn)
        else:
            self._generate(smiles=smiles, target=target, ids=ids)
            self._data = self._data.reshape(-1,1)
            self.save(self._cache_fn, overwrite=overwrite)

class FPolyDatasetV1(BaseDataset):
    def __init__(self, norm=True, **kwargs):
        super(FPolyDatasetV1, self).__init__(norm=norm, **kwargs)

    def generate(self,
                 path:str,
                 col_id:str = 'ID',
                 col_smiles:Union[List[str], str] = [f'SMILES_{x}' for x in 'ABCDE'],
                 col_weights:Union[List[str], str] = [f'FR_{x}' for x in 'ABCDE'],
                 col_target:Union[List[str], str] = ['TG'],
                 overwrite=False,
                 **kwargs):

        col_target = col_target if isinstance(col_target, List) else [col_target]
        self._target_desc = col_target
        df = pd.read_csv(path)
        df = df[np.sum(df[col_target].isna().values.reshape(-1, len(col_target)), axis=1) == 0]
        smiles = df[col_smiles]
        smiles = sorted(set(np.hstack(smiles.values[~smiles.isna()])))

        mask = [c in np.hstack([col_id, col_smiles, col_weights, col_target]).reshape(-1) for c in df.columns]

        self._tag = 'v1n' if self._norm else 'v1r'
        n_black = 0
        if isinstance(self._blacklist, dict):
            for v in self._blacklist.values():
                n_black += len(v)
        else:
            n_black = len(self._blacklist)
        if n_black != 0:
            self._tag += f'b{str(hex(n_black))}'
        self._tag += str(hex(int(''.join(np.array(mask).astype(int).astype(str).tolist()), 2))) 
        self._tag += '_' + str(hex(len(smiles))) + str(hex(df.shape[0]))
        print(self._tag)
        if self._check_cache(path) and not overwrite:
            self.load(self._cache_fn)
        else:
            self._generate(smiles=smiles)
                
            keydata = {d['smiles']:d for d in self._data}
            
            data = []
            pbar = None
            if df.shape[0] > 5000:
                pbar = tqdm.tqdm(desc='Parsing data', total=df.shape[0])
            for _, datarow in df.iterrows():
                id = datarow[col_id]
                weights = datarow[col_weights]
                m  = ~weights.isna()
                smiles = datarow[col_smiles].values[m]
                weights = torch.tensor(weights.values[m].astype(float)).float().view(-1)
                target = torch.tensor(datarow[col_target]).view(1,1).float()
                d = []
                w_sum = weights.sum()
                empty_data = self.empty_data.copy()
                empty_data.update({'id':id, 'target':target})
                for s, w in zip(smiles, weights):
                    data_dict = keydata[s].copy()
                    data_dict.update({
                        'id':id,
                        'weight':(w/w_sum).view(1,1).float(),
                        'target':target
                    })
                    d.append(data_dict)
                if len(d) == 0:
                    d.append(empty_data)
                data.append(d)
                if pbar is not None:
                    pbar.update(1)
            self._unique_data = to_numpy(np.array(self._data))
            self._data = np.array(data)
            self.save(self._cache_fn, overwrite=overwrite)
            self._data = to_tensor(self._data)

class FPolyDatasetV2(BaseDataset):
    def __init__(self, norm=True, **kwargs):
        super(FPolyDatasetV2, self).__init__(norm=norm, **kwargs)

    def generate(self,
                 path:str,
                 col_id:str = 'ID',
                 col_smiles:Union[List[str], str] = [f'SMILES_{x}' for x in 'ABCDE'],
                 col_weights:Union[List[str], str] = [f'FR_{x}' for x in 'ABCDE'],
                 col_target:Union[List[str], str] = ['TG'],
                 overwrite=False,
                 **kwargs):

        col_target = col_target if isinstance(col_target, List) else [col_target]
        self._target_desc = col_target
        
        df = pd.read_csv(path)
        df = df[np.sum(df[col_target].isna().values.reshape(-1, len(col_target)), axis=1) == 0]
        smiles = df[col_smiles]
        smiles = sorted(set(np.hstack(smiles.values[~smiles.isna()])))
        
        mask = [c in np.hstack([col_id, col_smiles, col_weights, col_target]).reshape(-1) for c in df.columns]

        self._tag = 'v2n' if self._norm else 'v2r'
        n_black = 0
        if isinstance(self._blacklist, dict):
            for v in self._blacklist.values():
                n_black += len(v)
        else:
            n_black = len(self._blacklist)
        if n_black != 0:
            self._tag += f'b{str(hex(n_black))}'
        self._tag += str(hex(int(''.join(np.array(mask).astype(int).astype(str).tolist()), 2))) 
        self._tag += '_' + str(hex(len(smiles))) + str(hex(df.shape[0]))

        if self._check_cache(path) and not overwrite:
            self.load(self._cache_fn)
        else:
            self._generate(smiles=smiles)
            
            pbar = None
            if df.shape[0] > 5000:
                pbar = tqdm.tqdm(desc='Parsing data', total=df.shape[0])

            keydata = {d['smiles']:d for d in self._data}
            data = []
            for _, datarow in df.iterrows():
                id = datarow[col_id]
                weights = datarow[col_weights]
                m  = ~ (weights.isna() | (weights == 0))
                smiles = datarow[col_smiles].values[m]
                weights = torch.tensor(weights.values[m].astype(float)).float().view(-1)
                target = torch.tensor(datarow[col_target]).view(1,-1).float()
                w_sum = weights.sum()
                f, c = [], []
                empty_data = self.empty_data.copy()
                empty_data.update({'id':id, 'target':target})
                for s, w in zip(smiles, weights):
                    data_dict = keydata[s].copy()
                    data_dict.update({
                        'id':id,
                        'weight':(w/w_sum).view(1,1).float(),
                        'target':target
                    })
                    if 'F' in s:
                        f.append(data_dict)
                    else:
                        c.append(data_dict)
                if len(f) == 0:
                    f.append(empty_data)
                if len(c) == 0:
                    c.append(empty_data)
                feat_f, _, _ = basic_collate_fn(f)
                feat_c, _, _ = basic_collate_fn(c)
                feat_f.update({'id':id, 'target':target, 'smiles':[s for s in smiles if 'F' in s]})
                feat_c.update({'id':id, 'target':target, 'smiles':[s for s in smiles if 'F' not in s]})
                data.append([feat_f, feat_c])
                if pbar is not None:
                    pbar.update(1)
            self._unique_data = to_numpy(np.array(self._data))
            self._data = np.array(data)
            self.save(self._cache_fn, overwrite=overwrite)
            self._data = to_tensor(self._data)

def PolyData(id, smiles, atom_feat, bond_feat, bond_idx, mol_feat, graph_idx, 
             target, weight=torch.ones((1,1)).float()):
        return {
            'id'       : id,        'smiles'  : smiles,   'atom_feat': atom_feat,
            'bond_feat': bond_feat, 'bond_idx': bond_idx, 'mol_feat' : mol_feat,
            'graph_idx': graph_idx, 'weight'   : weight,  'target'   : target,
        }
        
def to_tensor(dataset, device='cpu'):
    for data in dataset:
        if isinstance(data, dict):        
            for k, v in data.items():
                if k in ['id','smiles']:
                    continue
                if 'torch.Tensor' not in str(type(v)):
                    v = torch.tensor(v)
                data[k] = v.to(device)
        else:
            data = to_tensor(data, device=device)
    return dataset
    
def to_numpy(dataset):
    for data in dataset:
        if isinstance(data, dict):
            for k, v in data.items():
                if 'torch.Tensor' in str(type(v)):
                    v = v.cpu().numpy()
                    data[k] = v
        else:
            data = to_numpy(data)
    return dataset

def basic_collate_fn(dataset, device=None):
    ids       = []
    atom_feat = []
    bond_feat = []
    bond_idx  = []
    mol_feat  = []
    weight    = []
    target    = []
    graph_idx = []
    i_atom    = 0
    i_mol     = 0

    for data in dataset:
        ids.append(data['id'])
        atom_feat.append(data['atom_feat'])
        bond_feat.append(data['bond_feat'])
        bond_idx.append(data['bond_idx'] + i_atom)
        mol_feat.append(data['mol_feat'])
        graph_idx.append(data['graph_idx'] + i_mol)
        weight.append(data['weight'])
        target.append(data['target'])

        i_atom += data['atom_feat'].shape[0]
        i_mol  += data['mol_feat'].shape[0]

    atom_feats = torch.concat(atom_feat).float()
    bond_feats = torch.concat(bond_feat).float()
    bond_idxs  = torch.concat(bond_idx, axis=1).long()
    mol_feats  = torch.concat(mol_feat).float()
    weights    = torch.concat(weight).float()
    targets    = torch.concat(target).float()
    graph_idxs = torch.concat(graph_idx).long()

    if device is not None:
        atom_feats = atom_feats.to(device)
        bond_feats = bond_feats.to(device)
        bond_idxs  = bond_idxs.to(device)
        mol_feats  = mol_feats.to(device)
        weights    = weights.to(device)
        targets    = targets.to(device)
        graph_idxs = graph_idxs.to(device)

    features = {
        'atom_feat': atom_feats,
        'bond_feat': bond_feats,
        'bond_idx': bond_idxs,
        'mol_feat': mol_feats,
        'graph_idx': graph_idxs,
        'weight':weights,
    }
    return features, targets, np.array(ids).reshape(-1)

def collate_fn(dataset, device=None):
    feats = []
    for dset in np.array(dataset).T:
        feat, tgt, ids = basic_collate_fn(dset, device=device)
        idxs = [i * torch.ones(f['weight'].shape[0]) for i, f in enumerate(dset)]
        idxs = torch.hstack(idxs).long().to(tgt.device)
        feat.update({'data_idx':idxs})
        feats.append(feat)
    return feats, tgt, ids
    