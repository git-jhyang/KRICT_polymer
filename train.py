from src.utils.data import CrossValidation, train_test_split
from src.utils.dataset import FPolyDatasetV2, collate_fn
from src.utils.params import Parameters
from src.utils.functions import set_model, save_output, headers_epoch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
import os, argparse, json

parser = argparse.ArgumentParser(description=
    'Script for traing a neural network model with customizable parameters.',
    formatter_class=argparse.RawTextHelpFormatter
)

parser.add_argument('parameters', nargs='*', type=str, default=None, help=
    'Optional JSON files containing parameters to override the defaults. \n'
    'These paths are relative to the specified root directory unless an \n'
    'absolute path is provided. Files are processed in the order they are \n'
    'given, and if multiple files specify the same parameter, the value \n'
    'from the last file is used. Only include parameters in each file that \n'
    'you wish to change.'
)
parser.add_argument('--default', default='default.json',  type=str, help=
    'JSON file path containing default parameters for training a model.\n'
    'This path is relative to the specified root directory unless an \n'
    'absolute path is provided. Defaults to \'default.json\' within the\n' 
    'specified root directory.'
)

parser.add_argument('--root', type=str, default='./params/train', help=
    'Root directory where the parameter files are located. This is ignored \n'
    'if absolute paths are provided for \'default\' or \'parameter\' arguments. \n'
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
    train_ds.generate(path=p.train_data, col_id=p.id_column, col_smiles=p.smiles_column,
                      col_weights=p.weights_column, col_target=p.target_column)
    train_ds.to(p.device)
    
    test_ds = FPolyDatasetV2(p.normalize_feature, blacklist=p.blacklist, verbose=args.verbose)
    test_ds.generate(path=p.test_data, col_id=p.id_column, col_smiles=p.smiles_column,
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
    for n in range(num_repeat):
        model_desc = os.path.join(model_root, f'fold_{n:02d}')
        if os.path.isfile(os.path.join(model_desc, f'param.json')) and not p.overwrite:
            print('Target directory is not empty. Change \'overwrite = True\' in parameters or change directory. ', model_desc)
            continue
        os.makedirs(model_desc, exist_ok=True)
        writer = SummaryWriter(model_desc)
        if args.verbose:
            print('Model destination: ', model_desc)
            for h in headers_epoch: 
                print(h)

        if p.cross_valid:
            train_idx, valid_idx = cv[n]
        else:
            train_idx, valid_idx = train_test_split(train_ds, test_ratio=p.valid_ratio, return_index=True, seed=p.random_state+n)
        
        trainer, scheduler = set_model(p, train_ds, model_desc=model_desc, train_index=train_idx)
        train_dl = DataLoader(train_ds, batch_size=p.batch_size, sampler=SubsetRandomSampler(train_idx), collate_fn=collate_fn)
        valid_dl = DataLoader(train_ds, batch_size=512, sampler=valid_idx, collate_fn=collate_fn)

        for epoch in range(1, p.epochs + 1):
            _ = trainer.train(train_dl)            
            scheduler.step()
            
            if epoch % p.save_interval == 0:
                save_output(trainer, writer, f'{epoch:05d}', train_dl, valid_dl, test_dl, epoch, True, args.verbose)
            elif epoch % p.logging_interval == 0:
                save_output(trainer, writer, f'{epoch:05d}', train_dl, valid_dl, test_dl, epoch, False, args.verbose)
        if epoch % p.save_interval != 0:
            save_output(trainer, writer, f'{epoch:05d}', train_dl, valid_dl, test_dl, epoch, True, args.verbose)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)