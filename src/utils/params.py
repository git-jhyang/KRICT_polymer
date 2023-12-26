import torch, os, json

class Parameters:
    def __init__(self, fn=None, default='defaults.json', root='./params_pt'):
        default = default
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._load(os.path.join(root, default))
        if fn is not None:
            self._load(os.path.join(root, fn))
        
    def _load(self, fn):
        with open(fn) as f:
            d = json.load(f)
        for k,v in d.items():
            if hasattr(self, k) and isinstance(getattr(self, k), dict) and isinstance(v, dict):
                kv = getattr(self, k)
                kv.update(v)
                setattr(self, k, kv)
            else:
                setattr(self, k, v)
