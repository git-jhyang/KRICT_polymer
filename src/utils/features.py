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

RDMolDescList = [
    'MaxEStateIndex','MinEStateIndex','MaxAbsEStateIndex','MinAbsEStateIndex','qed','MolWt','HeavyAtomMolWt','ExactMolWt','NumValenceElectrons','NumRadicalElectrons','MaxPartialCharge','MinPartialCharge','MaxAbsPartialCharge','MinAbsPartialCharge','FpDensityMorgan1','FpDensityMorgan2','FpDensityMorgan3','BCUT2D_MWHI','BCUT2D_MWLOW','BCUT2D_CHGHI',
    'BCUT2D_CHGLO','BCUT2D_LOGPHI','BCUT2D_LOGPLOW','BCUT2D_MRHI','BCUT2D_MRLOW','BalabanJ','BertzCT','Chi0','Chi0n','Chi0v','Chi1','Chi1n','Chi1v','Chi2n','Chi2v','Chi3n','Chi3v','Chi4n','Chi4v','HallKierAlpha',
    'Ipc','Kappa1','Kappa2','LabuteASA','PEOE_VSA1','PEOE_VSA10','PEOE_VSA11','PEOE_VSA12','PEOE_VSA13','PEOE_VSA14','PEOE_VSA2','PEOE_VSA3','PEOE_VSA4','PEOE_VSA5','PEOE_VSA6','PEOE_VSA7','PEOE_VSA8','PEOE_VSA9','SMR_VSA1','SMR_VSA10',
    'SMR_VSA2','SMR_VSA3','SMR_VSA4','SMR_VSA5','SMR_VSA6','SMR_VSA7','SMR_VSA8','SMR_VSA9','SlogP_VSA1','SlogP_VSA10','SlogP_VSA11','SlogP_VSA12','SlogP_VSA2','SlogP_VSA3','SlogP_VSA4','SlogP_VSA5','SlogP_VSA6','SlogP_VSA7','SlogP_VSA8','SlogP_VSA9',
    'TPSA','EState_VSA1','EState_VSA10','EState_VSA11','EState_VSA2','EState_VSA3','EState_VSA4','EState_VSA5','EState_VSA6','EState_VSA7','EState_VSA8','EState_VSA9','VSA_EState1','VSA_EState10','VSA_EState2','VSA_EState3','VSA_EState4','VSA_EState5','VSA_EState6','VSA_EState7',
    'VSA_EState8','VSA_EState9','FractionCSP3','HeavyAtomCount','NHOHCount','NOCount','NumAliphaticCarbocycles','NumAliphaticHeterocycles','NumAliphaticRings','NumAromaticCarbocycles','NumAromaticHeterocycles','NumAromaticRings','NumHAcceptors','NumHDonors','NumHeteroatoms','NumRotatableBonds','NumSaturatedCarbocycles','NumSaturatedHeterocycles','NumSaturatedRings','RingCount',
    'MolLogP','MolMR','fr_Al_COO','fr_Al_OH','fr_Al_OH_noTert','fr_ArN','fr_Ar_COO','fr_Ar_N','fr_Ar_NH','fr_Ar_OH','fr_COO','fr_COO2','fr_C_O','fr_C_O_noCOO','fr_C_S','fr_HOCCN','fr_Imine','fr_NH0','fr_NH1','fr_NH2',
    'fr_N_O','fr_Ndealkylation1','fr_Ndealkylation2','fr_Nhpyrrole','fr_SH','fr_aldehyde','fr_alkyl_carbamate','fr_alkyl_halide','fr_allylic_oxid','fr_amide','fr_amidine','fr_aniline','fr_aryl_methyl','fr_azide','fr_azo','fr_barbitur','fr_benzene','fr_benzodiazepine','fr_bicyclic','fr_diazo',
    'fr_dihydropyridine','fr_epoxide','fr_ester','fr_ether','fr_furan','fr_guanido','fr_halogen','fr_hdrzine','fr_hdrzone','fr_imidazole','fr_imide','fr_isocyan','fr_isothiocyan','fr_ketone','fr_ketone_Topliss','fr_lactam','fr_lactone','fr_methoxy','fr_morpholine','fr_nitrile',
    'fr_nitro','fr_nitro_arom','fr_nitro_arom_nonortho','fr_nitroso','fr_oxazole','fr_oxime','fr_para_hydroxylation','fr_phenol','fr_phenol_noOrthoHbond','fr_phos_acid','fr_phos_ester','fr_piperdine','fr_piperzine','fr_priamide','fr_prisulfonamd','fr_pyridine','fr_quatN','fr_sulfide','fr_sulfonamd','fr_sulfone',
    'fr_term_acetylene','fr_tetrazole','fr_thiazole','fr_thiocyan','fr_thiophene','fr_unbrch_alkane','fr_urea','AUTOCORR2D_1','AUTOCORR2D_2','AUTOCORR2D_3','AUTOCORR2D_4','AUTOCORR2D_5','AUTOCORR2D_6','AUTOCORR2D_7','AUTOCORR2D_8','AUTOCORR2D_9','AUTOCORR2D_10','AUTOCORR2D_11','AUTOCORR2D_12','AUTOCORR2D_13',
    'AUTOCORR2D_14','AUTOCORR2D_15','AUTOCORR2D_16','AUTOCORR2D_17','AUTOCORR2D_18','AUTOCORR2D_19','AUTOCORR2D_20','AUTOCORR2D_21','AUTOCORR2D_22','AUTOCORR2D_23','AUTOCORR2D_24','AUTOCORR2D_25','AUTOCORR2D_26','AUTOCORR2D_27','AUTOCORR2D_28','AUTOCORR2D_29','AUTOCORR2D_30','AUTOCORR2D_31','AUTOCORR2D_32','AUTOCORR2D_33',
    'AUTOCORR2D_34','AUTOCORR2D_35','AUTOCORR2D_36','AUTOCORR2D_37','AUTOCORR2D_38','AUTOCORR2D_39','AUTOCORR2D_40','AUTOCORR2D_41','AUTOCORR2D_42','AUTOCORR2D_43','AUTOCORR2D_44','AUTOCORR2D_45','AUTOCORR2D_46','AUTOCORR2D_47','AUTOCORR2D_48','AUTOCORR2D_49','AUTOCORR2D_50','AUTOCORR2D_51','AUTOCORR2D_52','AUTOCORR2D_53',
    'AUTOCORR2D_54','AUTOCORR2D_55','AUTOCORR2D_56','AUTOCORR2D_57','AUTOCORR2D_58','AUTOCORR2D_59','AUTOCORR2D_60','AUTOCORR2D_61','AUTOCORR2D_62','AUTOCORR2D_63','AUTOCORR2D_64','AUTOCORR2D_65','AUTOCORR2D_66','AUTOCORR2D_67','AUTOCORR2D_68','AUTOCORR2D_69','AUTOCORR2D_70','AUTOCORR2D_71','AUTOCORR2D_72','AUTOCORR2D_73',
    'AUTOCORR2D_74','AUTOCORR2D_75','AUTOCORR2D_76','AUTOCORR2D_77','AUTOCORR2D_78','AUTOCORR2D_79','AUTOCORR2D_80','AUTOCORR2D_81','AUTOCORR2D_82','AUTOCORR2D_83','AUTOCORR2D_84','AUTOCORR2D_85','AUTOCORR2D_86','AUTOCORR2D_87','AUTOCORR2D_88','AUTOCORR2D_89','AUTOCORR2D_90','AUTOCORR2D_91','AUTOCORR2D_92','AUTOCORR2D_93',
    'AUTOCORR2D_94','AUTOCORR2D_95','AUTOCORR2D_96','AUTOCORR2D_97','AUTOCORR2D_98','AUTOCORR2D_99','AUTOCORR2D_100','AUTOCORR2D_101','AUTOCORR2D_102','AUTOCORR2D_103','AUTOCORR2D_104','AUTOCORR2D_105','AUTOCORR2D_106','AUTOCORR2D_107','AUTOCORR2D_108','AUTOCORR2D_109','AUTOCORR2D_110','AUTOCORR2D_111','AUTOCORR2D_112','AUTOCORR2D_113',
    'AUTOCORR2D_114','AUTOCORR2D_115','AUTOCORR2D_116','AUTOCORR2D_117','AUTOCORR2D_118','AUTOCORR2D_119','AUTOCORR2D_120','AUTOCORR2D_121','AUTOCORR2D_122','AUTOCORR2D_123','AUTOCORR2D_124','AUTOCORR2D_125','AUTOCORR2D_126','AUTOCORR2D_127','AUTOCORR2D_128','AUTOCORR2D_129','AUTOCORR2D_130','AUTOCORR2D_131','AUTOCORR2D_132','AUTOCORR2D_133',
    'AUTOCORR2D_134','AUTOCORR2D_135','AUTOCORR2D_136','AUTOCORR2D_137','AUTOCORR2D_138','AUTOCORR2D_139','AUTOCORR2D_140','AUTOCORR2D_141','AUTOCORR2D_142','AUTOCORR2D_143','AUTOCORR2D_144','AUTOCORR2D_145','AUTOCORR2D_146','AUTOCORR2D_147','AUTOCORR2D_148','AUTOCORR2D_149','AUTOCORR2D_150','AUTOCORR2D_151','AUTOCORR2D_152','AUTOCORR2D_153',
    'AUTOCORR2D_154','AUTOCORR2D_155','AUTOCORR2D_156','AUTOCORR2D_157','AUTOCORR2D_158','AUTOCORR2D_159','AUTOCORR2D_160','AUTOCORR2D_161','AUTOCORR2D_162','AUTOCORR2D_163','AUTOCORR2D_164','AUTOCORR2D_165','AUTOCORR2D_166','AUTOCORR2D_167','AUTOCORR2D_168','AUTOCORR2D_169','AUTOCORR2D_170','AUTOCORR2D_171','AUTOCORR2D_172','AUTOCORR2D_173',
    'AUTOCORR2D_174','AUTOCORR2D_175','AUTOCORR2D_176','AUTOCORR2D_177','AUTOCORR2D_178','AUTOCORR2D_179','AUTOCORR2D_180','AUTOCORR2D_181','AUTOCORR2D_182','AUTOCORR2D_183','AUTOCORR2D_184','AUTOCORR2D_185','AUTOCORR2D_186','AUTOCORR2D_187','AUTOCORR2D_188','AUTOCORR2D_189','AUTOCORR2D_190','AUTOCORR2D_191','AUTOCORR2D_192',
]
## descriptor functions
desc_fncs_ = {n:f for n,f in Descriptors.descList if n not in RDMolBlacklist}
desc_fncs_.update({'Ipc':lambda x: Descriptors.Ipc(x, avg=1)})
## 2D autocorrelation functions
desc_fncs_.update({f'AUTOCORR2D_{i}':eval(f'lambda x: Descriptors.AUTOCORR2D_{i}(x)') for i in range(1,193)})
desc_fncs = {n:desc_fncs_[n] for n in RDMolDescList}

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