import torch, os, json

class Parameters:
    def __init__(self, fn=None, default='defaults.json', root=''):
        default = default
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if os.path.isfile(default):
            self._load(default)
        else:
            self._load(os.path.join(root, default))
        if fn is not None:
            if os.path.isfile(fn):
                self._load(fn)
            else:
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
