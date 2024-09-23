import pandas as pd
from dgllife.utils import AttentiveFPAtomFeaturizer, AttentiveFPBondFeaturizer, Meter
from dgllife.data import MoleculeCSVDataset

def mydataset(csv_name, graph_name):
    df = pd.read_csv(csv_name)
    node_featurizer = AttentiveFPAtomFeaturizer(atom_data_field='feat')
    edge_featurizer = AttentiveFPBondFeaturizer(bond_data_field='feat')
    dataset = MoleculeCSVDataset(df,
                                 node_featurizer=node_featurizer,
                                 edge_featurizer=edge_featurizer,
                                 smiles_column='smiles',
                                 cache_file_path=graph_name)
    return dataset
