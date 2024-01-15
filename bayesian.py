from src.utils.data import CrossValidation, train_test_split
from src.utils.dataset import FPolyDatasetV2, collate_fn
from src.utils.params import Parameters
from src.utils.functions import set_model, save_output, headers_epoch, headers_bayes
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from bayes_opt import BayesianOptimization
import os, argparse, json, shutil

parser = argparse.ArgumentParser(description=
    'Script for traing a neural network model with customizable parameters. ',
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

parser.add_argument('--verbose', nargs='?', const=1, default=0, choices=[0,1,2], type=int, help=
    'Verbosity level of the output log. \n'
    '0: No verbose output. (default) \n'
    '1: Print summary of each Bayesian optimization step. (default if --verbose)\n'
    '2: Print detailed information, including the full training log over epochs.'
)

def optimization_function(lr):
    global num_iter, max_score, p, args, model_desc
    global train_idx, valid_dl, train_ds, test_dl
    
    path = os.path.join(model_desc, f'bayes_{num_iter:03d}')
    os.makedirs(path, exist_ok=True)

    writer = SummaryWriter(path)
    if args.verbose == 2:
        for h in headers_epoch: 
            print(h)
    
    p.learning_rate = lr
    trainer, scheduler = set_model(p=p, dataset=train_ds, model_desc=path, train_index=train_idx)
    train_dl = DataLoader(train_ds, batch_size=p.batch_size, sampler=SubsetRandomSampler(train_idx), collate_fn=collate_fn)
    
    for epoch in range(1, p.epochs + 1):
        _ = trainer.train(train_dl)
        scheduler.step()
    
        if epoch % p.save_interval == 0:
            output = save_output(trainer, writer, f'{epoch:05d}', train_dl, valid_dl, test_dl, epoch, True, args.verbose > 1)
        elif epoch % p.logging_interval == 0:
            output = save_output(trainer, writer, f'{epoch:05d}', train_dl, valid_dl, test_dl, epoch, False, args.verbose > 1)
    if epoch % p.save_interval != 0:
        output = save_output(trainer, writer, f'{epoch:05d}', train_dl, valid_dl, test_dl, epoch, True, args.verbose > 1)
    writer.close()
    
    out_train, out_valid, out_test = output
    _, train_r2, train_mae = out_train
    _, valid_r2, valid_mae = out_valid
    score = train_r2 + valid_r2 - p.bayes_mae_scale * (train_mae + valid_mae)
    with open(os.path.join(path, 'record.txt'),'w') as f:
        f.write(f'iter:{num_iter}\nlr:{lr}\nscore:{score}\n')
    if args.verbose == 2:
        for h in headers_bayes:
            print(h)
            
    # logging
    s = ['{:3d} | {:10.4e} | {:10.3e}'.format(num_iter, lr, score)]
    
    for _, r2, mae in [out_train, out_valid]:
        s.append('{:10.3e} {:10.4e}'.format(r2, mae))
    if out_test is not None:
        _, test_r2, test_mae = out_test
        s.append('{:10.3e} {:10.4e}'.format(test_r2, test_mae))
    log = ' | '.join(s)
    
    # evaluation
    if score < max_score * p.bayes_logging_tol:
        shutil.rmtree(path)
    if score > max_score:
        max_score = score
        for fn in ['model.torch','train.pkl','valid.pkl','test.pkl']:
            orig = os.path.join(path, f'{epoch:05d}.{fn}')
            if os.path.isfile(orig):
                shutil.copy(orig, os.path.join(model_desc, fn))
        for fn in ['param.json','scale.json','record.txt']:
            shutil.copy(os.path.join(path, fn), os.path.join(model_desc, fn))
        if args.verbose: 
            print(' *', log)
    elif args.verbose:
        print('  ', log)

    num_iter += 1
    return score

def main(args):
    global num_iter, max_score, p, model_desc
    global train_idx, valid_dl, train_ds, test_dl
    
    # load parameters
    params = [param for param in args.parameters if 'bayesian' not in param]
    args.parameters = ['./params/train/bayesian.json'] + list(params)
    p = Parameters(fns=args.parameters, default=args.default, root=args.root)
    if args.verbose:
        print(json.dumps(p.__dict__, indent=4))

    # dataset load
    train_ds = FPolyDatasetV2(p.normalize_feature, blacklist=p.blacklist, verbose=args.verbose)
    train_ds.generate(path=p.train_data, col_id=p.id_column, col_smiles=p.smiles_column,
                      col_weights=p.weights_column, col_target=p.target_column)
    train_ds.to(p.device)
    
    test_dl = None
    if hasattr(p, 'test_data'):
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
        if args.verbose > 0:
            print('Model destination: ', model_desc)
        if args.verbose == 1:
            for h in headers_bayes:
                print(h)
        
        if p.cross_valid:
            train_idx, valid_idx = cv[n]
        else:
            train_idx, valid_idx = train_test_split(train_ds, test_ratio=p.valid_ratio, return_index=True, seed=p.random_state+n)
        valid_dl = DataLoader(train_ds, batch_size=512, sampler=valid_idx, collate_fn=collate_fn)

        num_iter  = 0
        max_score = -1e8

        bo = BayesianOptimization(optimization_function, pbounds={'lr':p.bayes_lr_bound},
                                  random_state=p.random_state, allow_duplicate_points=True, verbose=0)
        bo.maximize(init_points=p.bayes_num_init, n_iter=p.bayes_num_iter)
        
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)