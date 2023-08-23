import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import BondType 
from rdkit.Chem import rdchem
from rdkit.Chem.rdchem import ChiralType

# atom_mapping
#átomos de caracter orgánico
atom_mapping = {"H":0,
                "B":1, 
                "C":2, 
                "N":3,
                "O":4, 
                "F":5, 
                "Si":6, 
                "P":7,
                "S":8,
                "Cl":9,
                "Se":10,
                "Br":11,
                "I":12
}
atom_mapping.update({0:"H",
                1:"B", 
                2:"C", 
                3:"N",
                4:"O", 
                5:"F", 
                6:"Si",
                7:"P", 
                8:"S",
                9:"Cl",
                10:"Se",
               11:"Br",
               12:"I"           
})
#chirality
chiral_mapping = {"CHI_TETRAHEDRAL_CW": 0, # R
          "CHI_TETRAHEDRAL_CCW":1, # S
          "CHI_UNSPECIFIED":2,
          "CHI_OTHER":3
                }
chiral_mapping.update({0: ChiralType.CHI_TETRAHEDRAL_CW, #R
              1: ChiralType.CHI_TETRAHEDRAL_CCW, #S
               2: ChiralType.CHI_UNSPECIFIED,
               3: ChiralType.CHI_OTHER  
              })

print(atom_mapping)

# bond_mapping
bond_mapping = ({"SINGLE":0, "DOUBLE":1, "TRIPLE":2, "AROMATIC":3})
bond_mapping.update({0: BondType.SINGLE, 1: BondType.DOUBLE,
                     2: BondType.TRIPLE, 3:BondType.AROMATIC
                      })
BATCH_SIZE = 100
EPOCHS = 10 #compute cycle

VAE_LR = 5e-4  #Velocidad de aprendizaje
MAX_MOLSIZE = 200 # Maximo número de átomos
MAX_ATOMS = 200

ATOMS = 13 #Número de tipos de átomos
BONDS = 4 + 1 #Número de tipos de enlaces
CHIRAL = 4 #Número de tipos de quiralidad
LATENT_DIM = 256 #Tamaño del espacio latente


def smiles_to_graph(smi):
  #Convertir SMILES  a objeto molecula
    mol = Chem.MolFromSmiles(smi)
  #Generar tensores de adjacency (enlaces entre átomos) y features (tipos de átomos)
  #adjacency matrices are always symmetric 
    CHIRAL = 3
    adjacency = np.zeros((BONDS, MAX_ATOMS, MAX_ATOMS), "float32") #5 x 120 x 120
  #print(adjacency.shape)
    features_a = np.zeros((MAX_ATOMS, ATOMS), "float32")  # 120 x 11 
    features_b = np.zeros((MAX_ATOMS, CHIRAL), "float32")  # 120 x 4 
  #print(features_a.shape, features_b.shape)

  #iterar sobre cada átomo (a) de cada molécula
    for a in mol.GetAtoms():
        i = a.GetIdx()  #átomo 1
        atom_type = atom_mapping[a.GetSymbol()] # DEVUELVE un valor númerico asignado al simbolo del átomo
        #print(atom_type)
        chiral_type = chiral_mapping[a.GetChiralTag().name]
        #print(chiral_type)
    
        features_a[i] = np.eye(ATOMS)[atom_type] #np.eye, devuelve una matriz con 1, codificada la posiion de UN ATOMO                                    
                                      # ATOMS, es la n-dimension de la matrix (nxn)
        features_b[i] = np.eye(CHIRAL)[chiral_type]
    
        features = np.concatenate([features_a, features_b], axis=1)
        # iterar sobre cada vecino
        for neighbor in a.GetNeighbors():
            j = neighbor.GetIdx()   #átomo 2
            bond = mol.GetBondBetweenAtoms(i, j)
            bond_type_idx = bond_mapping[bond.GetBondType().name] # VALOR NÚMERICO ASIGNADO AL TIPO DE ENLACE
            adjacency[bond_type_idx, [i, j], [j, i]] = 1
      
   
    return adjacency, features
    #return adjacency[-1,0]

def graph_to_molecule(graph):
    adjacency, features = graph
  #RWMol objeto molecula editable
    mol = Chem.RWMol()
  # Remover donde no haya atomos y atomos sin enlaces
 
    keep_idx = np.where(
      (np.argmax(features, axis=1) != ATOMS) 
  & (np.sum(adjacency[:-1], axis=(0,1)) != 0)
  )[0]
  #print(keep_idx)
    features = features[keep_idx]
    adjacency = adjacency[:, keep_idx, :][:, :, keep_idx]
  #print(features)

  # Adicionar atómos y quiralidad a la molécula
    for (atom_type_idx, chiral_idx) in zip(
      (np.argmax(features, axis=1)),
      (np.argmax(features[:, 13:], axis=1))):
        atom = Chem.Atom(atom_mapping[atom_type_idx]) # relacionar indices a número atómico 
        atom.SetChiralTag(chiral_mapping[chiral_idx]) #agregar quiralidad a los átomos de la molécula
        _ = mol.AddAtom(atom)
    
  # Adicionar enlaces entre los atomos en una molecula; basado en los triangulos superiores
  # Del tensor adjacente, simetrico
    (bonds_ij, atoms_i, atoms_j) = np.where(np.triu(adjacency) == 1)
    for (bond_ij, atom_i, atom_j) in zip(bonds_ij, atoms_i, atoms_j):
        if atom_i == atom_j or bond_ij == BONDS- 1:
            continue
        bond_type = bond_mapping[bond_ij]
        mol.AddBond(int(atom_i), int(atom_j), bond_type)

  # Sanitize the molecule; for more information on sanitization, see
   # https://www.rdkit.org/docs/RDKit_Book.html#molecular-sanitization
    flag = Chem.SanitizeMol(mol, catchErrors=True)
  # Let's be strict. If sanitization fails, return None
    if flag != Chem.SanitizeFlags.SANITIZE_NONE:
        return None
   
    return mol

