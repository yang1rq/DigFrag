import os
import argparse
import torch
import dgl
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MolFromSmiles, rdDepictor, Draw


def collate_molgraphs(data):
    smiles, graphs, labels, masks = map(list, zip(*data))

    bg = dgl.batch(graphs)
    labels = torch.stack(labels, dim=0)
    masks = torch.stack(masks, dim=0)

    return smiles, bg, labels, masks


def split_molecules_by_color(smiles, bg, atom_weight, threshold):
    att = atom_weight[0]

    min_value = torch.min(att)
    max_value = torch.max(att)
    att = (att - min_value) / (max_value - min_value)

    mol = Chem.MolFromSmiles(smiles[0])

    med_molecules = []
    pes_molecules = []

    for i in range(bg.number_of_nodes()):
        atom_color = att[i].data.item()
        if atom_color >= threshold:
            med_molecules.append(i)
        else:
            pes_molecules.append(i)

    med_molecule = Chem.RWMol(mol)
    for i in reversed(med_molecules):
        med_molecule.RemoveAtom(i)
    med_molecule = med_molecule.GetMol()

    pes_molecule = Chem.RWMol(mol)
    for i in reversed(pes_molecules):
        pes_molecule.RemoveAtom(i)
    pes_molecule = pes_molecule.GetMol()

    return [med_molecule], [pes_molecule]


def check_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in ['C', 'H', 'N', 'O', 'P', 'S', 'Br', 'Cl', 'I', 'F']:
            return False
    return True


def filter_mol(chem_name, filter_name):
    df = pd.read_csv(chem_name)
    df['mol'] = df['smiles'].apply(lambda x: Chem.MolFromSmiles(x))

    df = df.dropna(subset=['mol'])
    df['smiles'] = df['mol'].apply(lambda x: Chem.MolToSmiles(x))
    df = df.drop(labels='mol', axis=1)

    df = df[df['smiles'].str.len() < 128]
    df = df[df['smiles'].str.len() > 2]

    df = df[df['smiles'].apply(check_smiles)]
    df.to_csv(filter_name, index=False)


def draw_mol(frag_name, img_path):
    df = pd.read_csv(frag_name)

    if not os.path.exists(img_path):
        os.makedirs(img_path)
    for i, row in df.iterrows():
        smi = row['smiles']
        mol = Chem.MolFromSmiles(smi)
        Draw.MolToFile(mol, os.path.join(img_path, '{0}.png'.format(i)), size=(224, 224))


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--pred_path', type=str)
    parse.add_argument('--model_path', type=str)
    parse.add_argument('--frag_path', type=str)
    parse.add_argument('--image_path', type=str)
    return parse.parse_args()
