import torch, os, json

class Parameters:
    def __init__(self, fns=None, default='defaults.json', root=''):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if not default.endswith('.json'):
            default += '.json'
        if os.path.isfile(default):
            self._load(default)
        else:
            self._load(os.path.join(root, default))
        if fns is not None:
            tags = []
            for fn in fns:
                if not fn.endswith('.json'):
                    fn += '.json'
                if os.path.isfile(fn):
                    self._load(fn)
                    tags.append(self.tag)
                else:
                    self._load(os.path.join(root, fn))
                    tags.append(self.tag)
            self.tag = '_'.join(tags)
        
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
