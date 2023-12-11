import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.metrics import r2_score, mean_squared_error
from .base import reclusive_file_search

def read_data(path):
    s, t, p = pickle.load(open(path, 'rb'))
    order = np.argsort(s)
    return {
        'id':np.array(s)[order],
        'target':np.array(t).squeeze()[order],
        'pred':np.array(p).squeeze()[order],
    }

def read_best_from_subfolders(root, fn):
    fns = reclusive_file_search(root, fn)
    best_r2 = -1e9
    for fn in fns:
        d = read_data(fn)
        r2 = r2_score(d['target'], d['pred'])
        if r2 > best_r2:
            best_r2 = r2
            best_fn = fn
    return best_fn, read_data(best_fn)

def split_data(d):
    output = []
    for t, p in zip(d['target'].T, d['pred'].T):
        output.append({'id':d['id'], 'target':t, 'pred':p})
    return output

def read_cv_data(root, fn):
    output = {}
    output_fns = reclusive_file_search(root, fn)
    for output_fn in output_fns:
        tags = output_fn.split('/')[-2].split('_')
        if len(tags) != 4:
            continue
        tag = '.'.join(tags[:-1])
        data = read_data(output_fn)
        if tag not in output.keys():
            output[tag] = data
            output[tag]['pred'] = output[tag]['pred'].reshape(-1,1)
        else:
            id_ref = output[tag]['id']
            id_get = data['id']
            mask = id_ref != id_get
            if np.sum(mask) != 0:
                print('Warning: id mismatch', tag, output_fn)
                continue
            output[tag]['pred'] = np.hstack([output[tag]['pred'], data['pred'].reshape(-1,1)])
    return output

def screen_by_mask(data, mask):
    set_1 = {}
    set_2 = {}
    for key, val in data.items():
        set_1[key] = val[mask]
        set_2[key] = val[~mask]
    return set_1, set_2

def plot_scatter(ax, target, pred, xrange=None, metrics=['R2','RMSE','MAE'], 
                 fmts=['{:5.3f}','{:5.2f}','{:5.2f}'], unit='', **kwargs):
    if xrange is None:
        v_min = np.min([target.reshape(-1), pred.reshape(-1)])
        v_max = np.max([target.reshape(-1), pred.reshape(-1)])
        delta = (v_max - v_min)*0.2
    else:
        v_min, v_max = xrange
        delta = 0
    label = []
    for met, fmt in zip(metrics, fmts):
        if 'r2' in met.lower():
            r2 = r2_score(target, pred)
            lbl = 'R$^2$: < 0' if r2 < -1 else 'R$^2$: '+fmt.format(r2)
        elif 'rmse' in met.lower():
            lbl = 'RMSE: '+fmt.format(np.sqrt(mean_squared_error(target, pred)))
            lbl += f' {unit}'
        elif 'mae' in met.lower():
            lbl = 'MAE: '+fmt.format(np.mean(np.abs(target - pred)))
            lbl += f' {unit}'
        label.append(lbl)
    label = '\n'.join(label)
    ax.grid()
    ax.plot([v_min-delta, v_max+delta], [v_min-delta, v_max+delta], color=[0,0,0],
            ls='--', lw=1.5, zorder=30)
    ax.scatter(target, pred, **kwargs, label=label, zorder=15)
    ax.set_xlim([v_min-delta, v_max+delta])
    ax.set_ylim([v_min-delta, v_max+delta])
    leg = ax.legend(framealpha=1)
    leg.get_frame().set_facecolor([.95,.95,.95])
    leg.set_zorder(100)
    
    