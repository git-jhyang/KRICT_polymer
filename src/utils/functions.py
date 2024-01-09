from src.utils.data import DataScaler
from src.utils.dataset import collate_fn
from src.utils.trainer import Trainer
from src.model.modules import SingleEncoderModel
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import mean_absolute_error, r2_score
import torch, gc, os, pickle
import numpy as np

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

def save_output(trainer, path, pfx, train_dl, valid_dl=None, test_dl=None):
    trainer.model.save(path, model=f'{pfx}.model.torch')
    train_loss, id_train, tgt_train, prd_train = trainer.test(train_dl)
    train_r2 = [r2_score(t, p) for t,p in zip(tgt_train.T, prd_train.T)]
    train_mae = [mean_absolute_error(t, p) for t,p in zip(tgt_train.T, prd_train.T)]
    with open(os.path.join(path, f'{pfx}.train.pkl'), 'wb') as f:
        pickle.dump([id_train, tgt_train, prd_train], f)
    output_train = [train_loss, train_r2, train_mae]
    output_valid = None
    output_test = None
    if valid_dl is not None:
        valid_loss, id_valid, tgt_valid, prd_valid = trainer.test(valid_dl)
        valid_r2 = [r2_score(t, p) for t,p in zip(tgt_valid.T, prd_valid.T)]
        valid_mae = [mean_absolute_error(t, p) for t,p in zip(tgt_valid.T, prd_valid.T)]
        with open(os.path.join(path, f'{pfx}.valid.pkl'), 'wb') as f:
            pickle.dump([id_valid, tgt_valid, prd_valid], f)
        output_valid = [valid_loss, valid_r2, valid_mae]
    if test_dl is not None:
        test_loss, id_test, tgt_test, prd_test = trainer.test(test_dl)
        test_r2 = [r2_score(t, p) for t,p in zip(tgt_test.T, prd_test.T)]
        test_mae = [mean_absolute_error(t, p) for t,p in zip(tgt_test.T, prd_test.T)]
        with open(os.path.join(path, f'{pfx}.test.pkl'), 'wb') as f:
            pickle.dump([id_test, tgt_test, prd_test], f)
        output_test = [test_loss, test_r2, test_mae]
    return output_train, output_valid, output_test

