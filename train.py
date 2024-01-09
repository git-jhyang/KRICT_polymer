from src.utils.data import CrossValidation, train_test_split
from src.utils.dataset import FPolyDatasetV2, collate_fn
from src.utils.params import Parameters
from src.utils.functions import set_model, save_output
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import os, pickle, argparse

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

parser.add_argument('--root', type=str, default='./params/scratch', help=
    'Root directory where the parameter files are located. '
    'This is ignored if absolute paths are provided for \'default\' or \'parameter\' arguments. '
    'Defaults to \'./params/scratch\'. '
)

def main(args):
    # load parameters
    p = Parameters(fns=args.parameters, default=args.default, root=args.root)

    # dataset load
    ds = FPolyDatasetV2(p.normalize_feature, blacklist=p.blacklist)
    ds.generate(path=p.data_path, col_id=p)
    ds.to(p.device)
    
    data, test_data = train_test_split(ds, train_ratio=p.train_ratio, seed=p.random_state)
    test_dl = DataLoader(test_data, batch_size=p.batch_size, collate_fn=collate_fn)
    
    if p.cross_valid:
        cv = CrossValidation(n_fold = p.num_fold, data=data, seed=p.random_state, return_index=True)
        num_repeat = p.num_fold
    else:
        num_repeat = p.num_repeat

    # check directory
    model_root = os.path.join(p.output_path, p.encoder_type, p.tag)
    
    # training over loop
    for n in range(num_repeat):
        model_desc = os.path.join(model_root, f'n_{n:03d}')
        if os.path.isfile(os.path.join(model_desc, f'param.json')) and not p.overwrite:
            raise FileExistsError('Target directory is not empty. Change \'overwrite = True\' in parameters or change directory. ', model_desc)
        os.makedirs(model_desc, exist_ok=True)

        if p.cross_valid:
            train_idx, valid_idx = cv[n]
        else:
            train_idx, valid_idx = train_test_split(data, test_ratio=p.valid_ratio, return_index=True, seed=p.random_state+n)
        
        trainer, scheduler = set_model(p, data, model_desc=model_desc, train_index=train_idx)
        train_dl = DataLoader(data, batch_size=p.batch_size, sampler=SubsetRandomSampler(train_idx), collate_fn=collate_fn)
        valid_dl = DataLoader(data, batch_size=p.batch_size, sampler=valid_idx, collate_fn=collate_fn)

        for i in range(1, p.epochs + 1):
            loss = []
            outs = []
            train_loss = trainer.train(train_dl)
            scheduler.step()
            if i % 5 == 0:
                valid_loss, _, _, _ = trainer.test(valid_dl)
                loss.append([i, train_loss, valid_loss])
            if i % p.logging_interval == 0:
                out = save_output(trainer, model_desc, f'{i:05d}', train_dl, valid_dl, test_dl)
                outs.append(out)
            with open(os.path.join(model_desc, 'loss.pkl'), 'wb') as f:
                pickle.dump(loss, f)
            with open(os.path.join(model_desc, 'outs.pkl'), 'wb') as f:
                pickle.dump(outs, f)

    # training all data    
    if os.path.isfile(os.path.join(model_root, f'param.json')) and not p.overwrite:
        raise FileExistsError('Target directory is not empty. Change \'overwrite = True\' in parameters or change directory. ', model_root)
    os.makedirs(model_root, exist_ok=True)

    trainer, scheduler = set_model(p, data, model_desc=model_root, train_index=train_idx)
    train_dl = DataLoader(data, batch_size=p.batch_size, shuffle=True, collate_fn=collate_fn)

    for i in range(1, p.epochs + 1):
        loss = []
        outs = []
        train_loss = trainer.train(train_dl)
        scheduler.step()
        if i % 5 == 0:
            valid_loss, _, _, _ = trainer.test(valid_dl)
            loss.append([i, train_loss, valid_loss])
        if i % p.logging_interval == 0:
            out = save_output(trainer, model_root, f'{i:05d}', train_dl, valid_dl, test_dl)
            outs.append(out)
        with open(os.path.join(model_root, 'loss.pkl'), 'wb') as f:
            pickle.dump(loss, f)
        with open(os.path.join(model_root, 'outs.pkl'), 'wb') as f:
            pickle.dump(outs, f)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)