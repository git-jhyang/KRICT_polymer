import torch, os, json
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import (GATv2Conv, CGConv, 
                                TransformerConv, GMMConv, 
                                SplineConv, Sequential,
                                global_max_pool, global_mean_pool)
from typing import List

class BaseModel(nn.Module):
    '''
    Basic model class
    -----------
    Containing default save & load functions
    
    '''
    def __init__(self):
        super(BaseModel, self).__init__()
        self._params = {}
    
    def save(self, path:str, model:str, force=True):
        '''
        Saving pytorch model `torch.nn.module` object with model parameters in `param.json` 
        
        Parameters:
        -------------------
        `path` (`str`): The path to save model. \n
        `model` (`str`): The name of model. \n
        `force` (optional, `bool`): Whether to overwrite the existing model. (default: `True`)
        
        Returns:
        ---------------
        `None`
        '''
        os.makedirs(path, exist_ok=True)
        fn_model = os.path.join(path, model)
        fn_param = os.path.join(path, 'param.json')
        
        if os.path.isfile(fn_model) and not force:
            raise FileExistsError(fn_model)
        if os.path.isfile(fn_param) and not force:
            raise FileExistsError(fn_param)
        
        torch.save(self.state_dict(), fn_model)
        with open(fn_param, 'w') as f:
            json.dump(self._params, f, indent=4)
    
    def _load(self, cls, path:str, rebuild_model=False, requires_grad=False, 
              verbose=False, encoder_only=False, encoder_ID=None, **kwargs):
        '''
        Basic loading function. Loading saved model object with parameters
        
        Parameters
        ----------------
        `cls` (`torch.nn.Module`): The type of model to be loaded.\n
        `path` (`str`): The path to the saved model.\n
        `rebuild_model` (optional, `bool`): Whether to rebuild the model from scratch with loaded parameter. (default: `False`)\n
        `requires_grad` (optional, `bool`): Whether to calculate gradients for the loaded model parameters. (default: `False`)\n
        `verbose` (optional, `bool`): Whether to print out loading messages. (default: `False`)\n
        `encoder_only` (optional, `bool`): Whether to load only the encoder portion of the model. (default: `False`)\n
        `encoder_ID` (optional, `str`): The ID of the encoder to load, if applicable (`A` or `B`).\n
        **kwargs: Any additional keyword arguments to be passed to the model.

        Returns:
        --------------
        `cls` (`torch.nn.Module`): The loaded model.
        '''
        def _update_param(d_out, d_in):
            for k, v in d_in.items():
                if k not in d_out.keys():
                    d_out[k] = v
                elif isinstance(v, dict):
                    d_out[k] = _update_param(d_out[k], v)
                else:
                    d_out[k] = v
            return d_out
        
        if path.endswith('th') or path.endswith('torch'):
            fn_model = path
            fn_param = '/'.join(path.split('/')[:-1] + ['param.json'])
        else:
            fn_model = os.path.join(path, 'best.model.torch')
            fn_param = os.path.join(path, 'param.json')
        if not os.path.isfile(fn_model):
            raise FileNotFoundError(fn_model)
        if not os.path.isfile(fn_param):
            raise FileNotFoundError(fn_param)

        if rebuild_model:
            # rebuild model using the saved parameter file
            with open(fn_param) as f:
                _params = json.load(f)
            params = _update_param(self._params, _params)

            if encoder_only:
                # rebuild encoder only; Full model contains 1~5 encoders
                if 'encoder_params' in params.keys():
                    params = params['encoder_params']

            cls.__init__(**params) # rebuild model architecture
        
        model_dict   = cls.state_dict() # get current model state dict 
        trained_dict = torch.load(fn_model, map_location='cpu', **kwargs) # get saved model state dict
        if encoder_only:
            # extract encoder parameters only.
            if encoder_ID is not None and len([k for k in trained_dict.keys() if k.startswith(f'encoder_{encoder_ID}')]) != 0: 
                trained_dict = {k.replace(f'encoder_{encoder_ID}.',''):v for k,v in trained_dict.items() if k.startswith(f'encoder_{encoder_ID}')}
            else:
                trained_dict = {k.replace('encoder.',''):v for k,v in trained_dict.items() if 'encoder.' in k}
        trained_dict = {k:v for k,v in trained_dict.items() if k in model_dict.keys()}
        if verbose:
            # print loading state
            print('The following layers are transferred from the pretrained model:')
            print('\t', fn_model)
            print('-'*70)
            for k,v in trained_dict.items():
                print('\tName: ', k, '\t | Size: ', v.shape)
            print('-'*70)
        # update `model_dict` using `trained_dict`.
        model_dict.update(trained_dict)
        
        # update model parameters of `cls` saved in `model_dict`.
        cls.load_state_dict(model_dict)
        cls.requires_grad_(requires_grad=requires_grad)
        if hasattr(cls, 'decoder'):
            # calculate gradients of decoder
            cls.decoder.requires_grad_(True)
        return cls

    def load(self, path:str, requires_grad=False, rebuild_model=False, verbose=False, **kwargs):
        '''
        Default loading function
        
        Parameters
        ----------
        `path` (`str`): The path to the saved model.\n
        `rebuild_model` (optional, `bool`): Whether to rebuild the model from scratch with loaded parameter. (default: `False`)\n
        `requires_grad` (optional, `bool`): Whether to calculate gradients for the loaded model parameters. (default: `False`)\n
        `verbose` (optional, `bool`): Whether to print out loading messages. (default: `False`)\n
        **kwargs: Any additional keyword arguments to be passed to the model.
        
        Returns
        ------------
        `cls` (`torch.nn.Module`): The loaded model.
        '''
        return self._load(self, path=path, requires_grad=requires_grad, rebuild_model=rebuild_model, verbose=verbose, **kwargs)

    def load_encoder(self, path:str,requires_grad=False, rebuild_model=False, verbose=False, **kwargs):
        self.encoder = self._load(self.encoder, path=path, requires_grad=requires_grad, rebuild_model=rebuild_model, verbose=verbose, encoder_only=True, **kwargs)
        return self

class GraphNet(nn.Module):
    '''
    Basic graph neural network based on `torch.nn.Module` and `torch_geometric.nn` object.
    
    Parameters
    ----------
    `node_dim` (`int`): The dimensionality of the node features.\n
    `output_dim` (`int`): The dimensionality of the output features.\n
    `edge_dim` (`int`): The dimensionality of the edge features.\n
    `graph` (`str`, optional): The type of message passing graph that the module will operate on. (default: `CG`, one of `cg`, `gatv2`, `gmm`, `spline`, and `tf`)\n
    `hidden_dim` (`int`, optional): The dimensionality of hidden features. (default: 128)\n
    `n_layer` (`int`, optional): The number of layers. (default: 4)\n
    `readout` (`str`, optional): The type of readout operation to perform on the graph after the message passing. (default: `mean`, one of `mean` and `max`).\n
    
    Methods
    ----------
    `forward`: The forward pass of the neural network.
    
    Returns
    --------
    `None`
    '''
    def __init__(self, 
                 node_dim:int,
                 output_dim:int,
                 edge_dim:int,
                 graph:str = 'CG',
                 hidden_dim:int = 128,
                 n_layer = 4,
                 readout = 'mean',
                 ):
        super(GraphNet, self).__init__()
        
        # embedding layer
        self.graph_embed = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ELU(),
        )
        
        layers = []
        graph  = graph.lower()
        for _ in range(n_layer):
            # message passing layers, see https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#convolutional-layers
            if graph.lower() == 'cg':
                layer = CGConv(channels=hidden_dim, dim=edge_dim)
            elif graph.lower() == 'gatv2':
                layer = GATv2Conv(in_channels=hidden_dim, 
                                  out_channels=hidden_dim,
                                  edge_dim=edge_dim)
            elif graph.lower() == 'gmm':
                layer = GMMConv(in_channels=hidden_dim,
                                out_channels=hidden_dim,
                                dim=edge_dim,
                                kernel_size=8)
            elif graph.lower() == 'spline':
                layer = SplineConv(in_channels=hidden_dim,
                                   out_channels=hidden_dim,
                                   dim=edge_dim,
                                   kernel_size=8)
            elif graph.lower() == 'tf':
                layer = TransformerConv(in_channels=hidden_dim,
                                        out_channels=hidden_dim, 
                                        edge_dim=edge_dim)
            else:
                raise ValueError('not supported layer')
            layers.append((layer, 'x, edge_index, edge_attr -> x'))
            
        self.graph_network = Sequential('x, edge_index, edge_attr', layers) # see https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.sequential.Sequential
        self.graph_readout = eval(f'global_{readout}_pool') # see https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#pooling-layers
        self.graph_output = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, atom_feat, bond_idx, graph_idx, bond_feat=None):
        '''
        The forward pass of the neural network
        
        Parameters
        ----------
        `atom_feat` (`torch.Tensor`): The node features.
        `bond_idx` (`torch.Tensor`): The edge indices.
        `graph_idx` (`torch.Tensor`): The graph indices in the batch.
        `bond_feat` (`torch.Tensor`, optional): The edge features. (default: `None`)
        
        Returns
        -------
        `out` (`torch.Tensor`): The output features.
        
        Usage
        -----
        For the graph with `N` atoms and `M` bonds, the dimensionality of the parameters are given below:
        `atom_feat`: (`N`,  `node_dim`)
        `bond_idx`: (`2`, `M`)
        `graph_idx`: (`N`)
        `bond_feat`: (`M`, `edge_dim`)
        '''
        self._graph_embed = self.graph_embed(atom_feat)
        self._graph_network = self.graph_network(x=self._graph_embed, edge_index=bond_idx, edge_attr=bond_feat)
        self._graph_readout = self.graph_readout(self._graph_network, graph_idx)
        self._graph_output = self.graph_output(self._graph_readout)
        return self._graph_output


class MoleculeNet(nn.Module):
    '''
    Basic fully connected neural network based on `torch.nn.Module` object.
    
    Parameters
    ----------
    `input_dim` (`int`): The dimensionality of the input features.\n
    `output_dim` (`int`): The dimensionality of the output features.\n
    `hidden_dim` (`int`, optional): The dimensionality of hidden features. (default: 128)\n
    `n_layer` (`int`, optional): The number of layers. (default: 4)\n
    `activation` (`torch.nn.Module`, optional): The type of activation function. (default: `torch.nn.ELU()`).\n
    `dropout` (`float`, optional): The dropout ratio. (default: 0)

    Methods
    --------
    `forward`: The forward pass of the neural network.

    Returns
    --------
    `None`
    '''
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 hidden_dim = 256,
                 n_layer = 4,
                 activation:torch.nn.Module = nn.ELU(),
                 dropout = 0,
                 ):
        
        super(MoleculeNet, self).__init__()
        
        # embedding layer
        self.mol_embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
#            nn.BatchNorm1d(hidden_dim), 
            activation,
        )
        
        layers = []
        for _ in range(n_layer):
            # Linear layer
            layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
#                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(p=dropout),
                    activation
                )
            )
        self.mol_network = nn.Sequential(*layers)
        
        self.mol_output = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, mol_feat):
        '''
        The forward pass of the neural network
        
        Parameters
        ----------
        `mol_feat` (`torch.Tensor`): The vector features.
        
        Returns
        -------
        `out` (`torch.Tensor`): The output features
        '''
        self._mol_embed   = self.mol_embed(mol_feat)
        self._mol_network = self.mol_network(self._mol_embed)
        self._mol_output  = self.mol_output(self._mol_network)
        return self._mol_output


class MINE(nn.Module):
    '''
    MINE: Mutual Information Neural Estimator
    -----------------------------------------
    Estimate lower boundary of mutual information via neural network

    Paper: https://arxiv.org/abs/1801.04062
        
    Parameters
    ----------
    `input_dim` (`int`): The dimensionality of the input features. It should be the sum of the dimension of two input vectors.\n
    `hidden_dims` (`List[int, int]`, optional): The dimensionality of hidden features. (default: [256, 256])\n
    `activation` (`torch.nn.Module`, optional): The type of activation function. (default: `torch.nn.LeakyReLU`)

    Methods
    --------
    `forward`: The forward pass of the neural network.
    `loss`: The loss of MINE.

    Returns
    --------
    `None`
    '''
    def __init__(self, 
                 input_dim:int, 
                 hidden_dims:List[int] = [256, 256],
                 activation:torch.nn.Module = nn.LeakyReLU(negative_slope=0.2)
                 ):
        
        super(MINE, self).__init__()
        
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation)
            input_dim = hidden_dim        
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x1, x2):
        '''
        The forward pass of the neural network
        
        Parameters
        ----------
        `x1` (`torch.Tensor`): The first input vector.
        `x2` (`torch.Tensor`): The second input vector.
        
        Returns
        --------
        `mi` (`torch.Tensor`): The lower boundary of the mutual information between two vectors.
        '''
        #x2_shuffle = shuffle_sample(x2)
        x2_shuffle = torch.roll(x2, shifts=1, dims=0)
        
        pos = self.network(torch.cat([x1, x2], dim=-1))
        neg = self.network(torch.cat([x1, x2_shuffle], dim=-1))
        
        self._loss = F.softplus(-pos).mean() + F.softplus(neg).mean()
        
        mi_lb = pos.mean() - neg.exp().mean().log()
        
        return mi_lb
    
    @property
    def loss(self):
        '''
        The loss of MINE
        '''
        # Since MINE estimate lower boundary, it should be maximized -> negative sign of pos - neg
        return self._loss  

class CLUB(nn.Module):
    '''
    CLUB: Contrastive Log-ratio Upper Bound
    ---------------------------------------
    Estimate upper boundary of mutual information

    Paper: http://proceedings.mlr.press/v119/cheng20b
    Source: https://github.com/Linear95/CLUB/

    Parameters
    ----------
    `x1_dim` (`int`): The dimensionality of the first input features.\n
    `x2_dim` (`int`): The dimensionality of the second input features.\n
    `hidden_dims` (`List[int, int]`, optional): The dimensionality of hidden features. (default: [256, 256])\n
    `activation` (`torch.nn.Module`, optional): The type of activation function. (default: `torch.nn.LeakyReLU`)

    Methods
    --------
    `forward`: The forward pass of the neural network.
    `loss`: The loss of MINE.

    Returns
    --------
    `None`
    '''
    def __init__(self, 
                 x1_dim:int, 
                 x2_dim:int,
                 hidden_dims:List[int] = [256, 256],
                 activation:torch.nn.Module = nn.ReLU(),
                 ):
        super(CLUB, self).__init__()

        layers = []
        input_dim = x1_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation)
            input_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, x2_dim * 2))
        layers.append(nn.Tanh())

        self.gaussian = nn.Sequential(*layers)

    def forward(self, x1, x2):
        '''
        The forward pass of the neural network
        
        Parameters
        ----------
        `x1` (`torch.Tensor`): The first input vector, passing the gaussian network.
        `x2` (`torch.Tensor`): The second input vector.
        
        Returns
        --------
        `mi` (`torch.Tensor`): The upper boundary of the mutual information between two vectors.
        '''
        mu, logvar = torch.chunk(self.gaussian(x1), chunks=2, dim=-1)
        self._loss = - (- (mu - x2)**2 / logvar.exp() - logvar).sum(dim=-1).mean()
    
        pos = - 0.5 * (mu - x2)**2 / logvar.exp()
        neg = - 0.5 * ((mu.unsqueeze(1) - x2.unsqueeze(0))**2).mean(dim=1) / logvar.exp()
        
        mi_ub = (pos.sum(dim=-1) - neg.sum(dim=-1)).mean()
        return mi_ub

    @property
    def loss(self):
        '''
        log - likelyhood loss
        '''
        return self._loss


class HGIB(nn.Module):
    '''
    HGIB: Heterogeneous graph information bottleneck 
    ------------------------------------------------
    Maximize information between two embedded vectors, while discarding shared information.
    Paper: https://www.ijcai.org/proceedings/2021/0226.pdf
    
    Parameters
    ----------
    `encoder1` (`torch.nn.Module`): The first encoder.
    `encoder2` (`torch.nn.Module`): The second encoder.
    `x1_dim` (`int`): The dimensionality of the first input features.\n
    `x2_dim` (`int`): The dimensionality of the second input features.\n
    `hidden_dims` (`List[int, int]`, optional): The dimensionality of hidden features. (default: [256, 256])\n
    `beta` (`float`, optional): The balance between maximization and compression. (default: 1e-2)

    Methods
    --------
    `forward`: The forward pass of the neural network.
    `loss`: The loss of MINE.
    `scalars`: The detailed information.
    '''
    def __init__(self, 
                 encoder1, 
                 encoder2,
                 x1_dim:int,
                 x2_dim:int,
                 hidden_dims:List[int] = [256, 256], 
                 beta:float=1e-2):
        super(HGIB, self).__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self._beta = beta
        self.mi_estimator_0 = MINE(input_dim=x1_dim + x2_dim, hidden_dims=hidden_dims)
        self.mi_estimator_1 = CLUB(x1_dim=x1_dim, x2_dim=x2_dim, hidden_dims=hidden_dims)
        self.mi_estimator_2 = CLUB(x1_dim=x1_dim, x2_dim=x2_dim, hidden_dims=hidden_dims)
#        self.mi_estimator_1 = CLUBSample(x1_dim=x1_dim, x2_dim=x2_dim, hidden_dims=hidden_dims)
#        self.mi_estimator_2 = CLUBSample(x1_dim=x1_dim, x2_dim=x2_dim, hidden_dims=hidden_dims)

    def forward(self, x1, x2):
        '''
        The forward pass of the neural network
        
        Parameters
        ----------
        `x1` (`torch.Tensor`): The first input vector.
        `x2` (`torch.Tensor`): The second input vector.
        
        Returns
        --------
        `mi` (`torch.Tensor`): The lower boundary of the mutual information between two vectors.
        '''

        self._z1 = self.encoder1(**x1)
        self._z2 = self.encoder2(**x2)

        v1 = self._z1.mean(dim=0).expand_as(self._z1)
        v2 = self._z2.mean(dim=0).expand_as(self._z2)
        
        self._mi_z1z2 = self.mi_estimator_0(self._z1, self._z2)
        self._mi_v1z1 = self.mi_estimator_1(self._z1, v1)
        self._mi_v2z2 = self.mi_estimator_2(self._z2, v2)
        
        self._loss_z1z2 = self.mi_estimator_0.loss
        self._loss_v1z1 = self.mi_estimator_1.loss
        self._loss_v2z2 = self.mi_estimator_2.loss

        return self._mi_z1z2
    
    @property
    def loss(self):
        '''
        Total loss (MINE, CLUB_1, CLUB_2)
        '''
        return self._loss_z1z2 + self._beta * (self._loss_v1z1 + self._loss_v2z2)
        
    def embed(self, w1, w2, *args, **kwargs):
        return w1*self._z1 + w2*self._z2, self._z1, self._z2
    
    @property
    def scalars(self):
        '''
        Detailed information in dictionary.
        
        Keys
        ----
        `MI/Z1Z2`: The mutual information between `z1` and `z2`.
        `MI/V1Z1`: The mutual information between `v1` and `z1`.
        `MI/V2Z2`: The mutual information between `v2` and `z2`.
        `Loss/MINE`: The loss of `MINE`.
        `Loss/CLUBNet1`: The loss of `CLUB_1`.
        `Loss/CLUBNet2`: The loss of `CLUB_2`.
        '''
        return {
            'MI/Z1Z2':self._mi_z1z2,
            'MI/V1Z1':self._mi_v1z1,
            'MI/V2Z2':self._mi_v2z2,
            'Loss/MINE':self._loss_z1z2,
            'Loss/CLUBNet1':self._loss_v1z1,
            'Loss/CLUBNet2':self._loss_v2z2
        }