from src.utils.data import DataScaler
from src.utils.dataset import FPolyDatasetV3, collate_fn
from src.utils.trainer import Trainer
from src.model.modules import SingleEncoderModel
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import gc, torch, json, os, argparse

parser = argparse.ArgumentParser(description=
        '   This script processes a CSV file containing chemical data of copolymers and predict their glass transition\n'
        '   temperatures. Input CSV file should contains columns for ID, SMILES strings, and their corresponding ratios.',
    formatter_class=argparse.RawTextHelpFormatter
)

parser.add_argument('input_csv', type=str, help=
        'Path to the input CSV file. The file should contain columns for SMILES strings \n'
        'and ratio values, with each column labeled using the specified prefixes and suffix.\n\n'
        'example file format:\n'
        'ID,SMILES_A,SMILES_B,SMILES_C,SMILES_D,SMILES_E,FR_A,FR_B,FR_C,FR_D,FR_E\n'
        '0,C=C(C(=O)...,CC(=C)C...,CC(=C)C(O)=O,O=C1N(OC...,CC(=C)C...,2,2,2,2,2\n'
        '1,C=C(C(=O)O)C(F)(F)F,O=C1N(OC...,CC(=C)C(O)=O,CC(=C)C(=O)OCCO,,0.2,0.2,0.4,0.2,\n'
        '2,C=C(C(=O)O)C(F)(F)F,CC(=C)C(=O)OCCO,,,,10.4,2.5,,,'
)
parser.add_argument('--id', default='ID', type=str, help='Data ID column.')

parser.add_argument('--smiles_prefix', default='SMILES', type=str, help=
        'Prefix for SMILES columns in the CSV file. Defaults to "sm". This prefix will be \n'
        'combined with the suffix specified in --suffix to identify SMILES columns. \n'
        'Example from defaults are SMILES_A, SMILES_B, SMILES_C, etc.'
)
parser.add_argument('--ratio_prefix', default='FR', type=str, help=
        'Prefix for ratio columns in the CSV file. Defaults to "fr". This prefix will be \n'
        'combined with the suffix specified in --suffix to identify ratio columns. \n'
        'Example from defaults are FR_A, FR_B, FR_C, etc.'
)
parser.add_argument('--suffix', default='ABCDE', type=str, help=
        'Suffix to be added to the SMILES and ratio prefixes for identifying relevant \n'
        'columns in the CSV file, based on each individual character in the suffix. \n'
        'Recommand up to five character in sequence. \n'
        'For example, with a suffix "ABC", it will consider columns like SMILES_A, SMILES_B, SMILES_C.'
)

args = parser.parse_args()

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

ds = FPolyDatasetV3()
df = pd.read_csv(args.input_csv)
if args.id not in df.columns:
    df[args.id] = range(df.shape[0])

ds.generate(df, col_id=args.id, col_smiles=[f'{args.smiles_prefix}_{x}' for x in args.suffix], 
            col_weights=[f'{args.ratio_prefix}_{x}' for x in args.suffix], col_target=[args.id])
ds.to(device)
dl = DataLoader(ds, batch_size=512, collate_fn=collate_fn)

# loop over models
preds = []
for i in range(5):
    model = SingleEncoderModel(**json.load(open(os.path.join(model_path[i], 'param.json'))))
    model.load(os.path.join(model_path[i], 'model.torch'), rebuild_model=True)
    model.to(device)
    scaler = DataScaler(device=device)
    scaler.load(model_path[i])
    tr = Trainer(model, None, scaler)
    ids, pred = tr.predict(dl)
    preds.append(pred)
    
# final result
for i, p in zip(ids.reshape(-1), np.mean(preds, 0).reshape(-1)):
    print('    ID: {:5s} / Tg: {:8.3f} °C ({:.3f} K)'.format(str(i), p, p+273.15))