from src.utils.data import DataScaler
from src.utils.dataset import collate_fn
from src.utils.trainer import Trainer
from src.model.modules import SingleEncoderModel
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import mean_absolute_error, r2_score
import torch, gc, os, pickle
import numpy as np

bar = '-'*110
headers_epoch = [
    bar,
    ' Train | {:^32s} | {:^32s} | {:^32s}'.format('Train','Valid','Test'),
    ' epoch | {} | {} | {}'.format(*['{:>10s} {:>10s} {:>10s}'.format('Loss','R2','MAE')]*3),
    bar
]

headers_bayes = [
    bar,
    ' Bayes |   Learning |      Score | {:>21s} | {:>21s} | {:>21s}'.format('Train', 'Valid', 'Test'),
    '  iter |       rate |            | {} | {} | {}'.format(*['{:>10s} {:>10s}'.format('R2','MAE')]*3),
    bar,
]

def init_state(p):
    gc.collect()
    np.random.seed(p.random_state)
    torch.manual_seed(p.random_state)
    if p.device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.manual_seed(p.random_state)

def set_model(p, dataset, model_desc, train_index=None):
    init_state(p)
    # parameter load
    graph_net_params = p.graph_net_params.copy()
    mol_net_params = p.mol_net_params.copy()
    decoder_params = p.decoder_params.copy()
    graph_net_params.update({
        'node_dim':dataset.num_atom_feat,
        'edge_dim':dataset.num_bond_feat,    
    })
    mol_net_params.update({
        'input_dim':dataset.num_mol_feat,
    })
    encoder_params ={
        'graph_net_params' : graph_net_params,
        'mol_net_params' : mol_net_params,
    }
    decoder_params['output_dim'] = dataset.num_target

    # target scaling
    scaler = DataScaler(device=p.device)
    scaler.train(dataset, collate_fn=collate_fn, index=train_index)
    scaler.save(model_desc)

    model = SingleEncoderModel(encoder_type=p.encoder_type,
                               encoder_params=encoder_params, 
                               decoder_params=decoder_params)
    if not p.from_scratch:
        pt_path = os.path.join(p.pretrained_path, p.pretrained_model)
        if p.pretrained_encoder_only:
            model.load_encoder(pt_path, requires_grad=True)
        else:
            model.load(pt_path, requires_grad=True)
            scaler.load(pt_path)
    model.to(p.device)

    opt = AdamW(model.parameters(), lr=p.learning_rate)
    trainer = Trainer(model=model, opt=opt, scaler=scaler, noise_std=p.target_noise)
    scheduler = StepLR(optimizer=opt, step_size=p.scheduler_step_size, gamma=p.scheduler_gamma)
    return trainer, scheduler

def save_output(trainer, writer, pfx, train_dl, valid_dl=None, test_dl=None, 
                logging=False, saving=False, verbose=False):
    def _save_and_log_(dl, dl_name):
        loss, ids, tgt, pred = trainer.test(dl)
        r2s = [r2_score(t, p) for t, p in zip(tgt.T, pred.T)]
        maes = [mean_absolute_error(t, p) for t, p in zip(tgt.T, pred.T)]
        if logging:
            dlpfx = dl_name.capitalize()
            writer.add_scalar(f'{dlpfx}/Loss', loss, logging)
            for i, (r2, mae) in enumerate(zip(r2s, maes)):
                writer.add_scalar(f'{dlpfx}/R2_{i}', r2, logging)
                writer.add_scalar(f'{dlpfx}/MAE_{i}', mae, logging)
        if saving:
            with open(os.path.join(log_dir, f'{pfx}.{dl_name.lower()}.pkl'),'wb') as f:
                pickle.dump([ids, tgt, pred], f)
        return loss, np.mean(r2s), np.mean(maes)
    
    log_dir = writer.log_dir
    if saving:
        trainer.model.save(log_dir, model=f'{pfx}.model.torch')
    out_train = _save_and_log_(train_dl, 'train')
    out_valid = None
    out_test = None
    if valid_dl is not None:
        out_valid = _save_and_log_(valid_dl, 'valid')
    if test_dl is not None:
        out_test = _save_and_log_(test_dl, 'test')
    if verbose:
        s = [f'{logging:6d}']
        for out in [out_train, out_valid, out_test]:
            if out is None:
                s.append(' '*27)
            else:
                s.append('{:10.4e} {:10.3e} {:10.4e}'.format(*out))
        print(' | '.join(s))

    return out_train, out_valid, out_test