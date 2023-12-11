from mendeleev.fetch import fetch_table
from mendeleev import element
from rdkit import Chem
from rdkit.Chem import Descriptors
from typing import List
import numpy as np

def load_ptable_matrix(norm=True,
                       feature_names: List[str] = [], 
                       blacklist: List[str] = [],
                       **kwargs):
    '''
    Load periodic table from mendeleev

    Parameters
    ----------
    `norm` (`bool`, optional): Normalize periodic table by standard devistion. (default:`False`)\n
    `feature_names` (`List[str]`, optional): List of feature strings.\n
    `blacklist` (`List[str]`, optional): List of feature strings that will be excluded.
        
    Returns
    -------
    `features` (`np.ndarray`): Periodic table.\n
    `feature_names` (`List[str]`): Column names.
    '''
    if len(feature_names) == 0:
        feature_names = _ptable_feature_names.copy()
    feature_names = [fn for fn in feature_names if fn not in blacklist]
    ptable = fetch_table('elements')[feature_names]
    ele_configs = np.zeros((ptable.shape[0], 0), dtype=float)
    
    if 'block' in feature_names:
        ptable.loc[ptable.loc[:, 'block'] == 's', 'block'] = 0.0
        ptable.loc[ptable.loc[:, 'block'] == 'p', 'block'] = 1.0
        ptable.loc[ptable.loc[:, 'block'] == 'd', 'block'] = 2.0
        ptable.loc[ptable.loc[:, 'block'] == 'f', 'block'] = 3.0
    if 'electronic_configuration' in feature_names:
        ptable_ec = ptable.pop('electronic_configuration')
        ele_configs = np.zeros((ptable.shape[0], 4), dtype=float)
        for i, config in ptable_ec.items():
            for orbit in config.split():
                if '[' in orbit: continue
                for j, qn in enumerate('spdf'):
                    if qn not in orbit: continue
                    _, k = orbit.split(qn)
                    if k == '': k = 1
                    ele_configs[i, j] += int(k)
        feature_names.pop(feature_names.index('electronic_configuration'))
        feature_names.extend([f'ele_config_{c}' for c in 'spdf'])
    atom_feats = np.nan_to_num(np.hstack([np.array(ptable, dtype=float), ele_configs])[:96, :])
    ion_engs = np.zeros((atom_feats.shape[0], 1))

    for i in range(0, ion_engs.shape[0]):
        ion_eng = element(i + 1).ionenergies

        if 1 in ion_eng:
            ion_engs[i, 0] = ion_eng[1]
        else:
            ion_engs[i, 0] = 0

    features = np.hstack((atom_feats, ion_engs))
    feature_names.append('ion_energy')
    if norm: features = features / np.std(features, axis=0)
    return features, feature_names

class RDKitMoleculeDescriptor:
    '''
    RDKit 2D descriptors

    Parameters
    ----------
    `norm` (`bool`, optional): Normalize periodic table by standard devistion. (default:`False`)\n
    `include_autocorr` (`bool`, optional): Include auto-correlation functions. (default:`True`)\n
    `blacklist` (`List[str]`): List of feature strings that will be excluded.
    
    Properties
    ----------
    `descriptors` (`List[str]`): List of features.
    
    Methods
    -------
    `get`: Get molecule descriptors from given RDKit molecule.\n
    '''

    def __init__(self, 
                 norm = True, 
                 include_autocorr = True, 
                 blacklist:List[str] = [], 
                 **kwargs):
        
        self._norm = norm
        self._blacklist = blacklist
        desc_dict = {n:fnc for n, fnc in desc_fncs.items() if n not in self._blacklist}
        if not include_autocorr:
            desc_dict = {n:fnc for n, fnc in desc_dict.items() if 'AUTOCORR2D' not in n}
        
        self._desc_fncs  = []
        self._desc_names = []
        for name, fnc in desc_dict.items():
            if self._norm:
                if name in RDMolLogNorm:
                    fnc = lambda x: np.log(fnc(x)) if fnc(x) > 1 else 0
                if name in RDMolScaleNorm.keys():
                    scale = RDMolScaleNorm[name]
                    fnc = lambda x: fnc(x) * scale
            self._desc_names.append(name)
            self._desc_fncs.append(fnc)
 
    def get(self, m):
        '''
        Parameters
        ---------------
        `m` (`rdkit.Chem.rdchem.Mol`): RDKit molecule.

        Returns
        -----------------
        `feats` (`np.ndarray`): Set of molecular feature. (`1, N_feat`) shape matrix.
        '''
        feats = []
        for fnc in self._desc_fncs:
            feats.append(fnc(m))
        return np.array(feats)[np.newaxis, :]

    @property
    def descriptors(self):
        return self._desc_names


class RDKitAtomicGraphDescriptor:
    '''
    RDKit graph descriptors

    Parameters
    ----------
    `norm` (`bool`, optional): Normalize periodic table by standard devistion. (default:`False`)\n
    `blacklist` (`List[str]`): List of feature strings that will be excluded.
    
    Properties
    ----------
    `atom_descriptors` (`List[str]`): List of atom features.
    `bond_descriptors` (`List[str]`): List of bond features.
    
    Methods
    -------
    `get`: Get graph descriptors from given RDKit molecule.\n
    '''
    def __init__(self, 
                 include_ptable = True,
                 blacklist: List[str] = [], 
                 **kwargs):

        self._blacklist = blacklist
        self._include_ptable = include_ptable
        self._atom_fncs = {k:f for k,f in atom_fncs.items() if k not in self._blacklist}
        self._bond_fncs = {k:f for k,f in bond_fncs.items() if k not in self._blacklist}
        
        self._atom_feat_names = [k for k in self._atom_fncs.keys()]
        self._bond_feat_names = [k for k in self._bond_fncs.keys()]
        
        if self._include_ptable:
            self._ptable, _ptable_feat_names = load_ptable_matrix(blacklist=self._blacklist, **kwargs)
            self._atom_feat_names = np.hstack([self._atom_feat_names, _ptable_feat_names]).tolist()
            
    def get(self, m):
        '''
        Parameters
        ---------------
        `m` (`rdkit.Chem.rdchem.Mol`): RDKit molecule

        Returns
        -----------------
        `atom_feats` (`np.ndarray`): Set of atom features. (`N_atom`, `N_atom_feat`) shape matrix.\n
        `bond_feats` (`np.ndarray`): Set of bond features. (`N_bond`, `N_bond_feat`) shape matrix.\n
        `nbr_idxs` (`np.ndarray`): Set of atom indices of corresponing bonds. (`2`, `N_bond`) shape indices.
        '''
        atom_nums  = []
        atom_feats = []
        bond_feats = []
        nbr_idxs   = []
        for atom in m.GetAtoms():
            atom_nums.append(atom.GetAtomicNum())
            atom_feats.append([fnc(atom) for fnc in self._atom_fncs.values()])
        for bond in m.GetBonds(): 
            ii = bond.GetBeginAtomIdx()
            ij = bond.GetEndAtomIdx()
            bond_feat = [fnc(bond) for fnc in self._bond_fncs.values()]
            # bidirectional message passing
            nbr_idxs.append((ii, ij))
            bond_feats.append(bond_feat)
            nbr_idxs.append((ij, ii))
            bond_feats.append(bond_feat)

        if 'IsRotatable' not in self._blacklist:
            rot_bonds = m.GetSubstructMatches(RotatableStructure)
            for i, idx in enumerate(nbr_idxs):
                if idx not in rot_bonds:
                    continue
                bond_feats[i][-1] = 1
        
        atom_nums = np.array(atom_nums, dtype=int) - 1
        atom_feats = np.array(atom_feats, dtype=float)
        if self._include_ptable:
            atom_feats = np.hstack([atom_feats, self._ptable[atom_nums]])
        if len(m.GetBonds()) != 0:
            bond_feats = np.array(bond_feats, dtype=float)
            nbr_idxs = np.array(nbr_idxs, dtype=int).T
        else:
            bond_feats = np.zeros((0,len(self._bond_fncs)), dtype=float)
            nbr_idxs = np.zeros((2,0), dtype=int)
        return atom_feats, bond_feats, nbr_idxs

    @property
    def atom_descriptors(self):
        return self._atom_feat_names
    
    @property
    def bond_descriptors(self):
        return self._bond_feat_names
    
# default ptable feature names
_ptable_feature_names = [
    'atomic_number', 'atomic_volume', 'block', 'covalent_radius_pyykko',
    'electron_affinity', 'electronic_configuration', 'en_pauling', 'fusion_heat',
    'metallic_radius', 'vdw_radius_bondi', 'period'
]

# rotatable bond structure
RotatableStructure = Chem.MolFromSmarts('[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]')

# RDKit descriptor functions
## blacklist
RDMolBlacklist = [
    'Kappa3',
]
## descriptor functions
desc_fncs = {n:f for n,f in Descriptors.descList if n not in RDMolBlacklist}
desc_fncs.update({'Ipc':lambda x: Descriptors.Ipc(x, avg=1)})
## 2D autocorrelation functions
desc_fncs.update({f'AUTOCORR2D_{i}':eval(f'lambda x: Descriptors.AUTOCORR2D_{i}(x)') for i in range(1,193)})

# RDKit descriptor normalizer - too large values are scaled for NN
## log normalizer, truncate lower than 1, then np.log(f(x))
RDMolLogNorm = [
    'MolWt', 'HeavyAtomMolWt','ExactMolWt', 'MaxPartialCharge', 'NumValenceElectrons',
    'BCUT2D_MWHI', 'BCUT2D_MWLOW', 'BertzCT', 'LabuteASA', 'MolMR',     
]
## scale normalizer
RDMolScaleNorm = {'TPSA':1e-2}
RDMolScaleNorm.update({n:1e-2 for n in desc_fncs.keys() if 'VSA' in n})
RDMolScaleNorm.update({f'AUTOCORR2D_{i}':1e-2 for i in range(89,97)})

# RDKit graph feature functions
atom_fncs = {
#    'AtomicNum': lambda x: x.GetAtomicNum(),
    'Mass': lambda x: x.GetMass(),
    'Degree': lambda x: x.GetDegree(),
#    'FormalCharge': lambda x: x.GetFormalCharge(),
    'ImplicitValence': lambda x: x.GetImplicitValence(),
    'ExplicitValence': lambda x: x.GetExplicitValence(),
    'Hybridization': lambda x: x.GetHybridization(),
    'IsAromatic': lambda x: x.GetIsAromatic(),
#    'Isotope': lambda x: x.GetIsotope(),
#    'NoImplicit': lambda x: x.GetNoImplicit(),
    'NumImplicitHs': lambda x: x.GetNumImplicitHs(),
    'NumExplicitHs': lambda x: x.GetNumExplicitHs(),
#    'NumRadicalElectrons': lambda x: x.GetNumRadicalElectrons(),
    'TotalDegree': lambda x: x.GetTotalDegree(),
    'TotalNumHs': lambda x: x.GetTotalNumHs(),
    'TotalValence': lambda x: x.GetTotalValence(),
}

bond_fncs = {
#    'BeginAtomIdx': lambda x: x.GetBeginAtomIdx(),
#    'EndAtomIdx': lambda x: x.GetEndAtomIdx(),
#    'BondDir': lambda x: x.GetBondDir(),
    'BondType': lambda x: x.GetBondType(),
    'BondTypeAsDouble': lambda x: x.GetBondTypeAsDouble(),
    'IsAromatic': lambda x: x.GetIsAromatic(),
    'IsConjugated': lambda x: x.GetIsConjugated(),
    'IsRotatable': lambda _: 0,
}