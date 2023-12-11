import os

def copy_doc(inp):
    '''
    Copy docstring
    '''
    def wrapper(tgt):
        tgt.__doc__ = inp.__doc__
        return tgt
    return wrapper


def reclusive_file_search(root:str, fn:str):
    fns = []
    for dir in os.listdir(root):
        path = os.path.join(root, dir)
        if os.path.isdir(path):
            fns.extend(reclusive_file_search(root=path, fn=fn))
        else:
            break
    if fn in os.listdir(root):
        fns.append(os.path.join(root, fn))
    return fns
