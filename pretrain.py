from src.utils.data import DataScaler, CrossValidation, train_test_split
from src.utils.dataset import QM9Dataset, collate_fn
from src.utils.trainer import Trainer
from src.utils.params import Parameters
from src.model.modules import SingleEncoderModel
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import mean_absolute_error, r2_score
import torch, gc, os, pickle, argparse
import numpy as np

parser = argparse.ArgumentParser(description=
    'Pretraining script for model training with customizable parameters.',
    formatter_class=argparse.RawTextHelpFormatter
)

parser.add_argument('parameters', nargs='?', type=str, default=None, help=
    'Optional JSON file containing parameters to override the defaults. '
    'This path is relative to the specified root directory unless an absolute path is provided. '
    'Only include parameters that you wish to change.'                    
)
parser.add_argument('--default', default='default.json',  type=str, help=
    'JSON file path containing default parameters for pretraining. '
    'This path is relative to the specified root directory unless an absolute path is provided. '
    'Defaults to \'default.json\' within the specified root directory.'
)

parser.add_argument('--root', type=str, default='./params/pretrain', help=
    'Root directory where the parameter files are located. '
    'This is ignored if absolute paths are provided for \'default\' or \'parameter\' arguments. '
    'Defaults to \'./params/pretrain\'. '

)

args = parser.parse_args()

def main():
    global args

    # load parameters
    p = Parameters(fn=args.parameters, default=args.default, root=args.root)

    # check directory
    model_desc = os.path.join(p.output_path, p.encoder_type, p.tag)
    if os.path.isfile(os.path.join(model_desc, f'param.json')) and not p.overwrite:
        raise FileExistsError('Target directory is not empty. Change \'overwrite = True\' in parameters or change directory. ', model_desc)

    scaler = DataScaler(device=p.device)




if __name__ == '__main__':
    main()