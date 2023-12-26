from src.utils.data import DataScaler
from src.utils.dataset import FPolyDatasetV3, collate_fn
from src.utils.trainer import Trainer
from src.model.modules import SingleEncoderModel
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import gc, torch, json, os

# Input part / up to N monomer & feed ratio pairs
data = [
    [ # first dataset
        ['C=C(C(=O)O)C(F)(F)F', 0.2], # first monomer and feed ratio
        ['CC(=C)C(=O)OC12CC3CC(C1)CC(C3)C2', 0.2], # second monomer and feed ratio
        ['CC(=C)C(O)=O', 0.2],
        ['O=C1N(OC(=O)C(=C)C)C(=O)CC1', 0.2],
        ['CC(=C)C(=O)OCCO', 0.2],
    ],
    [ # second dataset
        ['C=C(C(=O)O)C(F)(F)F', 0.2], # first monomer and feed ratio
        ['O=C1N(OC(=O)C(=C)C)C(=O)CC1', 0.2], # second monomer and feed ratio
        ['CC(=C)C(O)=O', 0.4],
        ['CC(=C)C(=O)OCCO', 0.2],
    ]
]

# parameters
device = 'cpu'
model_path = [
    './outputs/finetune/all_ens/fold_00',
    './outputs/finetune/all_ens/fold_01',
    './outputs/finetune/all_ens/fold_02',
    './outputs/finetune/all_ens/fold_03',
    './outputs/finetune/all_ens/fold_04',
]

# Data part

gc.collect()
torch.cuda.empty_cache()

n = np.max([len(l) for l in data])
for d in data:
    for _ in range(n-len(d)):
        d.append([np.nan, np.nan])

df = pd.concat([
    pd.DataFrame(range(len(data)), columns=['ID']),
    pd.DataFrame([[_d[0] for _d in d] for d in data], columns=[f'sm_{i}' for i in range(n)]),
    pd.DataFrame([[_d[1] for _d in d] for d in data], columns=[f'fr_{i}' for i in range(n)])
], axis=1)

DS = FPolyDatasetV3()
DS.generate(df, col_id='ID', col_smiles=[f'sm_{i}' for i in range(n)], 
            col_weights=[f'fr_{i}' for i in range(n)], col_target=['ID'])
DS.to(device)
DL = DataLoader(DS, batch_size=512, collate_fn=collate_fn)

# loop over models
preds = []
for i in range(5):
    model = SingleEncoderModel(**json.load(open(os.path.join(model_path[i], 'param.json'))))
    model.load(os.path.join(model_path[i], 'model.torch'), rebuild_model=True)
    model.to(device)
    scaler = DataScaler(device=device)
    scaler.load(model_path[i])
    tr = Trainer(model, None, scaler)
    ids, pred = tr.predict(DL)
    preds.append(pred)

# final result
print(np.mean(preds, 0).reshape(-1))