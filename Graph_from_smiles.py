from typing import Any

import torch

import torch_geometric
import numpy as np
x_map = {
    'atomic_num':
    list(range(0, 119)), # Atomic number between 1-118
    'chirality': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
        'CHI_TETRAHEDRAL',
        'CHI_ALLENE',
        'CHI_SQUAREPLANAR',
        'CHI_TRIGONALBIPYRAMIDAL',
        'CHI_OCTAHEDRAL',
    ],
    'degree':
    list(range(0, 11)),
    'formal_charge':
    list(range(-2, 2)),
    'num_hs':
    list(range(0, 9)),
    'num_radical_electrons':
    list(range(0, 5)),
    'hybridization': [
        'UNSPECIFIED',
        'S',
        'SP',
        'SP2',
        'SP3',
        'SP3D',
        'SP3D2',
        'OTHER',
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}

e_map = {
    'bond_type': [
        'UNSPECIFIED',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'QUADRUPLE',
        'QUINTUPLE',
        'HEXTUPLE',
        'ONEANDAHALF',
        'TWOANDAHALF',
        'THREEANDAHALF',
        'FOURANDAHALF',
        'FIVEANDAHALF',
        'AROMATIC',
        'IONIC',
        'HYDROGEN',
        'THREECENTER',
        'DATIVEONE',
        'DATIVE',
        'DATIVEL',
        'DATIVER',
        'OTHER',
        'ZERO',
    ],
    'stereo': [
        'STEREONONE',
        'STEREOANY',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
    ],
    'is_conjugated': [False, True],
}


def from_smiles(y_val,
                smiles: str, with_hydrogen: bool = False,
                kekulize: bool = False,
                ) -> 'torch_geometric.data.Data':
    r"""Converts a SMILES string to a :class:`torch_geometric.data.Data`
    instance.

    Args:
        smiles (str): The SMILES string.
        with_hydrogen (bool, optional): If set to :obj:`True`, will store
            hydrogens in the molecule graph. (default: :obj:`False`)
        kekulize (bool, optional): If set to :obj:`True`, converts aromatic
            bonds to single/double bonds. (default: :obj:`False`)
    """
    from rdkit import Chem, RDLogger

    from torch_geometric.data import Data

    RDLogger.DisableLog('rdApp.*')

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        mol = Chem.MolFromSmiles('')
    if with_hydrogen:
        mol = Chem.AddHs(mol)
    if kekulize:
        Chem.Kekulize(mol)

    xs = []
    for atom in mol.GetAtoms():
        x = []
        x.append(x_map['atomic_num'].index(atom.GetAtomicNum())) # 1
        x.append(x_map['chirality'].index(str(atom.GetChiralTag()))) # 2
        x.append(x_map['degree'].index(atom.GetTotalDegree())) # 3
        x.append(x_map['formal_charge'].index(atom.GetFormalCharge())) #4
        x.append(x_map['num_hs'].index(atom.GetTotalNumHs())) #5
        x.append(x_map['num_radical_electrons'].index(
            atom.GetNumRadicalElectrons()))  # 6
        x.append(x_map['hybridization'].index(str(atom.GetHybridization())))  #7
        x.append(x_map['is_aromatic'].index(atom.GetIsAromatic()))  # 8
        x.append(x_map['is_in_ring'].index(atom.IsInRing()))  # 9
        xs.append(x)

    x = torch.tensor(xs, dtype=torch.long).view(-1, 9)
    #x = torch.tensor(xs, dtype=torch.float32).view(-1,9)
                    
    #print(x)
    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        e = []
        e.append(e_map['bond_type'].index(str(bond.GetBondType())))
        e.append(e_map['stereo'].index(str(bond.GetStereo())))
        e.append(e_map['is_conjugated'].index(bond.GetIsConjugated()))

        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]

    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.long).view(2, -1)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 3)
    #edge_attr = torch.tensor(edge_attrs, dtype=torch.float32).view(-1,3)

    if edge_index.numel() > 0:  # Sort indices.
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

    # construct label tensor


    #convert NPL-score value to one-hot class vector
    def val_to_class(Y_val):
        if Y_val < -1: #synthetic compound
            return [1, 0, 0]
        elif Y_val < 0: #slightly natural product
            return [0, 1, 0]
        else: # Natural Product
            return [0, 0, 1]


    y_tensor = torch.tensor(np.array([y_val]), dtype = torch.float)
    #y_tensor = torch.tensor([val_to_class(Y_val)], dtype=torch.float)
    print(y_tensor)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y = y_tensor, smiles=smiles)
