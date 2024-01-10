from src.utils.data import CrossValidation, train_test_split
from src.utils.dataset import FPolyDatasetV2, collate_fn
from src.utils.params import Parameters
from src.utils.functions import set_model, save_output
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import os, pickle, argparse, json
import numpy as np

parser = argparse.ArgumentParser(description=
    'Script for traing a neural network model with customizable parameters.',
    formatter_class=argparse.RawTextHelpFormatter
)

parser.add_argument('parameters', nargs='*', type=str, default=None, help=
    'Optional JSON files containing parameters to override the defaults. '
    'These paths are relative to the specified root directory unless an absolute '
    'path is provided. Files are processed in the order they are given, and if '
    'multiple files specify the same parameter, the value from the last file is used. '
    'Only include parameters in each file that you wish to change.'
)
parser.add_argument('--default', default='default.json',  type=str, help=
    'JSON file path containing default parameters for training a model. '
    'This path is relative to the specified root directory unless an absolute path is provided. '
    'Defaults to \'default.json\' within the specified root directory.'
)

parser.add_argument('--root', type=str, default='./params/train', help=
    'Root directory where the parameter files are located. '
    'This is ignored if absolute paths are provided for \'default\' or \'parameter\' arguments. '
    'Defaults to \'./params/train\'. '
)

parser.add_argument('--verbose', default=False, action='store_true', help=
    'Print training log. Defaults to False.'
)

def main(args):
    # load parameters
    p = Parameters(fns=args.parameters, default=args.default, root=args.root)
    if args.verbose:
        print(json.dumps(p.__dict__, indent=4))
    # dataset load
    train_ds = FPolyDatasetV2(p.normalize_feature, blacklist=p.blacklist, verbose=args.verbose)
    train_ds.generate(path=p.data_path, col_id=p.id_column, col_smiles=p.smiles_column,
                      col_weights=p.weights_column, col_target=p.target_column)
    train_ds.to(p.device)
    
    test_ds = FPolyDatasetV2(p.normalize_feature, blacklist=p.blacklist, verbose=args.verbose)
    test_ds.generate(path=p.data_path, col_id=p.id_column, col_smiles=p.smiles_column,
                     col_weights=p.weights_column, col_target=p.target_column)
    test_ds.to(p.device)
    test_dl = DataLoader(test_ds, batch_size=512, collate_fn=collate_fn)
    
    if p.cross_valid:
        cv = CrossValidation(n_fold = p.num_fold, data=train_ds, seed=p.random_state, return_index=True)
        num_repeat = p.num_fold
    else:
        num_repeat = p.num_repeat

    # check directory
    model_root = os.path.join(p.output_path, p.tag)
    
    # training over loop
    interval = np.max([np.round(p.epochs / 20), 1])
    for n in range(num_repeat):
        model_desc = os.path.join(model_root, f'fold_{n:02d}')
        if os.path.isfile(os.path.join(model_desc, f'param.json')) and not p.overwrite:
            print('Target directory is not empty. Change \'overwrite = True\' in parameters or change directory. ', model_desc)
            continue
        os.makedirs(model_desc, exist_ok=True)
        if args.verbose:
            print('Model destination: ', model_desc)
            print(' Epoch / Train loss / Valid loss')
        if p.cross_valid:
            train_idx, valid_idx = cv[n]
        else:
            train_idx, valid_idx = train_test_split(train_ds, test_ratio=p.valid_ratio, return_index=True, seed=p.random_state+n)
        
        trainer, scheduler = set_model(p, train_ds, model_desc=model_desc, train_index=train_idx)
        train_dl = DataLoader(train_ds, batch_size=p.batch_size, sampler=SubsetRandomSampler(train_idx), collate_fn=collate_fn)
        valid_dl = DataLoader(train_ds, batch_size=512, sampler=valid_idx, collate_fn=collate_fn)

        for epoch in range(1, p.epochs + 1):
            loss = []
            outs = []
            train_loss = trainer.train(train_dl)
            valid_loss, _, _, _ = trainer.test(valid_dl)
            loss.append([epoch, train_loss, valid_loss])
            if epoch % interval == 0:
                if args.verbose:
                    print('{:6d} / {:10.6f} / {:10.6f}'.format(epoch, train_loss, valid_loss))
            scheduler.step()
            if epoch % p.logging_interval == 0:
                out = save_output(trainer, model_desc, f'{epoch:05d}', train_dl, valid_dl, test_dl)
                outs.append(out)
        if epoch % p.logging_interval != 0:
            out = save_output(trainer, model_desc, f'{epoch:05d}', train_dl, valid_dl, test_dl)
            outs.append(out)
        np.savetxt(os.path.join(model_desc, 'loss.txt'), loss, fmt=['%6d','%10.6f','%10.6f'])
        with open(os.path.join(model_desc, 'outs.pkl'), 'wb') as f:
            pickle.dump(outs, f)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)