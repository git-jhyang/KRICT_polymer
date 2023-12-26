import torch
from torch import nn
from .base import BaseModel, GraphNet, MoleculeNet, HGIB, MINE, CLUB
from ..utils.base import copy_doc
from torch_geometric.nn import global_add_pool, global_max_pool
import numpy as np

class GraphEncoder(BaseModel):
    '''
    Graph encoder
    
    Parameters
    ----------
    `graph_net_params` (`dict`): The set of graph neural network parameters.
     - `node_dim` (`int`): The dimensionality of the node features.
     - `output_dim` (`int`): The dimensionality of the output features.
     - `edge_dim` (`int`): The dimensionality of the edge features.
     - `graph` (`str`, optional): The type of message passing graph that the module will operate on. (default: `CG`, one of `cg`, `gatv2`, `gmm`, `spline`, and `tf`)
     - `hidden_dim` (`int`, optional): The dimensionality of hidden features. (default: 128)
     - `n_layer` (`int`, optional): The number of layers. (default: 4)
     - `readout` (`str`, optional): The type of readout operation to perform on the graph after the message passing. (default: `mean`, one of `mean` and `max`).\n

    Methods
    ----------
    `forward`: The forward pass of the neural network.
    '''
    def __init__(self, graph_net_params, **kwargs):
        super(GraphEncoder, self).__init__()
        self._params = {
            'graph_net_params':graph_net_params,
        }
        self.graphnet = GraphNet(**graph_net_params)
        self.output_dim = graph_net_params['output_dim']

    @copy_doc(GraphNet.forward)
    def forward(self, atom_feat, bond_feat, bond_idx, graph_idx, **kwargs):
        out = self.graphnet(atom_feat=atom_feat, 
                            bond_feat=bond_feat,
                            bond_idx=bond_idx,
                            graph_idx=graph_idx)
        return out

class MoleculeEncoder(BaseModel):
    '''
    Molecule encoder
    
    Parameters
    ----------
    `mol_net_params` (`dict`): The set of dense neural network parameters.
     - `atom_feat` (`torch.Tensor`): The node features.
     - `bond_idx` (`torch.Tensor`): The edge indices.
     - `graph_idx` (`torch.Tensor`): The graph indices in the batch.
     - `bond_feat` (`torch.Tensor`, optional): The edge features. (default: `None`)

    Methods
    ----------
    `forward`: The forward pass of the neural network.
    '''
    def __init__(self, mol_net_params, **kwargs):
        super(MoleculeEncoder, self).__init__()
        self._params = {
            'mol_net_params':mol_net_params
        }
        self.molnet     = MoleculeNet(**mol_net_params)
        self.output_dim = mol_net_params['output_dim']
    
    @copy_doc(MoleculeNet.forward)
    def forward(self, mol_feat, **kwargs):
        out = self.molnet(mol_feat=mol_feat)
        return out

class ConcatEncoder(BaseModel):
    '''
    Concatenated encoder of GraphNet and MoleculeNet
    
    Parameters
    ----------
    `graph_net_params` (`dict`): The set of graph neural network parameters.
     - `node_dim` (`int`): The dimensionality of the node features.
     - `output_dim` (`int`): The dimensionality of the output features.
     - `edge_dim` (`int`): The dimensionality of the edge features.
     - `graph` (`str`, optional): The type of message passing graph that the module will operate on. (default: `CG`, one of `cg`, `gatv2`, `gmm`, `spline`, and `tf`)
     - `hidden_dim` (`int`, optional): The dimensionality of hidden features. (default: 128)
     - `n_layer` (`int`, optional): The number of layers. (default: 4)
     - `readout` (`str`, optional): The type of readout operation to perform on the graph after the message passing. (default: `mean`, one of `mean` and `max`).\n

    `mol_net_params` (`dict`): The set of dense neural network parameters.
     - `atom_feat` (`torch.Tensor`): The node features.
     - `bond_idx` (`torch.Tensor`): The edge indices.
     - `graph_idx` (`torch.Tensor`): The graph indices in the batch.
     - `bond_feat` (`torch.Tensor`, optional): The edge features. (default: `None`)

    Methods
    ----------
    `forward`: The forward pass of the neural network.
    '''

    def __init__(self, graph_net_params, mol_net_params, **kwargs):
        super(ConcatEncoder, self).__init__()
        if graph_net_params['output_dim'] != mol_net_params['output_dim']:
            dim = np.max([graph_net_params['output_dim'], mol_net_params['output_dim']])
            graph_net_params['output_dim'] = dim
            mol_net_params['output_dim'] = dim
        self._params = {
            'graph_net_params':graph_net_params,
            'mol_net_params':mol_net_params
        }
        self.graphnet = GraphNet(**graph_net_params)
        self.molnet   = MoleculeNet(**mol_net_params)
        self.output_dim = graph_net_params['output_dim'] + mol_net_params['output_dim']
        
    def forward(self, atom_feat, mol_feat, bond_idx, graph_idx, bond_feat=None, **kwargs):
        '''
        The forward pass of the neural network
        
        Parameters
        ----------
        `mol_feat` (`torch.Tensor`): The vector features.
        `atom_feat` (`torch.Tensor`): The node features.
        `bond_idx` (`torch.Tensor`): The edge indices.
        `graph_idx` (`torch.Tensor`): The graph indices in the batch.
        `bond_feat` (`torch.Tensor`, optional): The edge features. (default: `None`)
        
        Returns
        -------
        `out` (`torch.Tensor`): The output features
        
        Usage
        -----
        For the graph with `N` atoms and `M` bonds, the dimensionality of the parameters are given below:
        `atom_feat`: (`N`,  `node_dim`)
        `bond_idx`: (`2`, `M`)
        `graph_idx`: (`N`)
        `bond_feat`: (`M`, `edge_dim`)
        '''
        h1 = self.graphnet(atom_feat=atom_feat, 
                           bond_feat=bond_feat,
                           bond_idx=bond_idx,
                           graph_idx=graph_idx)
        h2 = self.molnet(mol_feat=mol_feat)
        out = torch.cat([h1, h2], dim=-1)
        return out
        
class LinearDecoder(BaseModel):
    '''
    Fully connected neural network decoder
    
    Parameters
    ----------
    `decoder_params` (`dict`): The set of dense neural network parameters.
     - `input_dim` (`int`): The dimensionality of the input features.
     - `output_dim` (`int`): The dimensionality of the output features.
     - `hidden_dim` (`int`): The dimensionality of hidden features. 
     - `n_layer` (`int`): The number of layers.

    Methods
    ----------
    `forward`: The forward pass of the neural network.
    
    Returns
    --------
    `None`

    '''
    def __init__(self, decoder_params):
        super(LinearDecoder, self).__init__()
        self._params = {
            'decoder_params':decoder_params
        }
        input_dim = decoder_params['input_dim']
        output_dim = decoder_params['output_dim']
        hidden_dim = decoder_params['hidden_dim']
        n_layer = decoder_params['n_layer']
        
        self.embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.2)
        )
        layers = []
        for i in range(n_layer):
            layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
#                    nn.BatchNorm1d(hidden_dim),
                    nn.LeakyReLU(negative_slope=0.2)
                )
            )
        self.network = nn.Sequential(*layers)
        self.linear_output = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        '''
        The forward pass of the neural network
        
        Parameters
        ----------
        `x` (`torch.Tensor`): The input features.

        Returns
        -------
        `out` (`torch.Tensor`): The output features.
        '''
        h = self.embed(x)
        h = self.network(h)
        out = self.linear_output(h)
        return out

class SingleEncoderModel(BaseModel):
    '''
    Regression model for single molecule.
    
    Parameters
    -------------------------
    `encoder_type` (`str`): The type of encoder architecture.
    `encoder_params` (`dict[dict, dict]`): The set of encoder parameters.
     - `graph_net_params` (`dict`): The set of graph neural network parameters.
       - `node_dim` (`int`): The dimensionality of the node features.
       - `output_dim` (`int`): The dimensionality of the output features.
       - `edge_dim` (`int`): The dimensionality of the edge features.
       - `graph` (`str`, optional): The type of message passing graph that the module will operate on. (default: `CG`, one of `cg`, `gatv2`, `gmm`, `spline`, and `tf`)
       - `hidden_dim` (`int`, optional): The dimensionality of hidden features. (default: 128)
       - `n_layer` (`int`, optional): The number of layers. (default: 4)
       - `readout` (`str`, optional): The type of readout operation to perform on the graph after the message passing. (default: `mean`, one of `mean` and `max`).\n
     - `mol_net_params` (`dict`): The set of dense neural network parameters.
       - `atom_feat` (`torch.Tensor`): The node features.
       - `bond_idx` (`torch.Tensor`): The edge indices.
       - `graph_idx` (`torch.Tensor`): The graph indices in the batch.
       - `bond_feat` (`torch.Tensor`, optional): The edge features. (default: `None`)\n
    `decoder_params` (`dict`): The set of decoder parameters.
       - `input_dim` (`int`): The dimensionality of the input features.
       - `output_dim` (`int`): The dimensionality of the output features.
       - `hidden_dim` (`int`): The dimensionality of hidden features. 
       - `n_layer` (`int`): The number of layers.

    Methods
    ----------
    `forward`: The forward pass of the neural network.

    '''
    def __init__(self, encoder_type, encoder_params, decoder_params, **kwargs):
        super(SingleEncoderModel, self).__init__()
        self._params = {
            'encoder_type':encoder_type,
            'encoder_params':encoder_params,
            'decoder_params':decoder_params,
        }
        if 'graph' in encoder_type.lower():
            self.encoder = GraphEncoder(**encoder_params)
        elif 'mol' in encoder_type.lower():
            self.encoder = MoleculeEncoder(**encoder_params)
        elif 'cat' in encoder_type.lower():
            self.encoder = ConcatEncoder(**encoder_params)
        else:
            raise ValueError('not supported encoder type:', encoder_type)
        decoder_params['input_dim'] = self.encoder.output_dim * 2
#        decoder_params['input_dim'] = self.encoder.output_dim
        self.decoder = LinearDecoder(decoder_params)
    
    def forward(self, feats, *args, **kwargs):
        h1s = []
        h2s = []
        bs  = []
        for feat in feats:
            w = feat['weight']
            b = feat['data_idx']
            h = self.encoder(**feat)
#            hs.append(torch.hstack([global_add_pool(h * w, b), global_add_pool(h, b)]))
            h1s.append(global_add_pool(h * w, b))
            h2s.append(global_add_pool(h, b))
            bs.append(global_add_pool(torch.ones_like(b).view(-1,1), b))
        self._embd = torch.hstack([
            torch.stack(h1s, dim=0).sum(dim=0),
            torch.stack(h2s, dim=0).sum(dim=0) / torch.stack(bs, dim=0).sum(dim=0).view(-1,1),
        ])
        out = self.decoder(self._embd)
        return out

class DualEncoderModel(BaseModel):
    def __init__(self, encoder_type, encoder_params, decoder_params, **kwargs):
        
        super(DualEncoderModel, self).__init__()
        
        self._params = {
            'encoder_type':encoder_type,
            'encoder_params':encoder_params,
            'decoder_params':decoder_params,
        }
        
        if 'graph' in encoder_type.lower():
            self.encoder_A = GraphEncoder(**encoder_params)
            self.encoder_B = GraphEncoder(**encoder_params)
        elif 'mol' in encoder_type.lower():
            self.encoder_A = MoleculeEncoder(**encoder_params)
            self.encoder_B = MoleculeEncoder(**encoder_params)
        elif 'cat' in encoder_type.lower():
            self.encoder_A = ConcatEncoder(**encoder_params)
            self.encoder_B = ConcatEncoder(**encoder_params)
        else:
            raise ValueError('not supported encoder type:', encoder_type)    
        decoder_params['input_dim'] = self.encoder_A.output_dim * 2
        self.decoder = LinearDecoder(decoder_params)

    def load_encoder(self, path:str, requires_grad=False, rebuild_model=False, verbose=False, **kwargs):
        self.encoder_A = self._load(self.encoder_A, path=path, requires_grad=requires_grad, 
                                    rebuild_model=rebuild_model, verbose=verbose, 
                                    encoder_only=True, encoder_ID='A', **kwargs)
        self.encoder_B = self._load(self.encoder_B, path=path, requires_grad=requires_grad, 
                                    rebuild_model=rebuild_model, verbose=verbose, 
                                    encoder_only=True, encoder_ID='B', **kwargs)
        return self

    def forward(self, feats, *args, **kwargs):
        hs = []
        for encoder, feat in zip([self.encoder_A, self.encoder_B], feats):
            w = feat['weight']
            b = feat['data_idx']
            h = encoder(**feat)
            hs.append(torch.hstack([global_add_pool(h * w, b), global_add_pool(h, b)]))
        self._embd = torch.stack(hs, dim=0).sum(dim=0)
        out = self.decoder(self._embd)
        return out

class IMaxEncoderModel(BaseModel):
    def __init__(self, encoder_type, encoder_params, beta=1e-3, **kwargs):
        
        super(IMaxEncoderModel, self).__init__()

        self.beta = beta
        self._params = {
            'encoder_type':encoder_type,
            'encoder_params':encoder_params,
            'beta':self.beta,
        }
        
        if 'graph' in encoder_type.lower():
            self.encoder_A = GraphEncoder(**encoder_params)
            self.encoder_B = GraphEncoder(**encoder_params)
        elif 'mol' in encoder_type.lower():
            self.encoder_A = MoleculeEncoder(**encoder_params)
            self.encoder_B = MoleculeEncoder(**encoder_params)
        elif 'cat' in encoder_type.lower():
            self.encoder_A = ConcatEncoder(**encoder_params)
            self.encoder_B = ConcatEncoder(**encoder_params)
        else:
            raise ValueError('not supported encoder type:', encoder_type)

        self.mine = MINE(self.encoder_A.output_dim * 4)
        self.club_A = CLUB(self.encoder_A.output_dim * 2, self.encoder_B.output_dim * 2)
        self.club_B = CLUB(self.encoder_A.output_dim * 2, self.encoder_B.output_dim * 2)

    def load_encoder(self, path:str, requires_grad=False, rebuild_model=False, verbose=False, **kwargs):
        self.encoder_A = self._load(self.encoder_A, path=path, requires_grad=requires_grad, 
                                    rebuild_model=rebuild_model, verbose=verbose, encoder_only=True, **kwargs)
        self.encoder_B = self._load(self.encoder_B, path=path, requires_grad=requires_grad, 
                                    rebuild_model=rebuild_model, verbose=verbose, encoder_only=True, **kwargs)
        return self

    def forward(self, feats, **kwargs):
        hs = []
        for encoder, feat in zip([self.encoder_A, self.encoder_B], feats):
            w = feat['weight']
            b = feat['data_idx']
            h = encoder(**feat)
            hs.append(torch.hstack([global_add_pool(h * w, b), global_add_pool(h, b)]))
        h1, h2 = hs

        v1 = h1.mean(dim=0).expand_as(h1)
        v2 = h2.mean(dim=0).expand_as(h2)
        
        self.mi_h1h2 = self.mine(h1, h2)
        self.mi_v1h1 = self.club_A(v1, h1)
        self.mi_v2h2 = self.club_B(v2, h2)
        
        return self.mi_h1h2
        
    @property
    def loss(self):
        return self.mine.loss + self.beta * (self.club_A.loss + self.club_B.loss)

    @property
    def scalars(self):
        return {
            'MI/MINE': self.mi_h1h2,  'Loss/MINE': self.mine.loss,
            'MI/CLUB_A': self.mi_v1h1,  'Loss/CLUB_A': self.club_A.loss * self.beta,
            'MI/CLUB_B': self.mi_v2h2,  'Loss/CLUB_B': self.club_B.loss * self.beta,
        }