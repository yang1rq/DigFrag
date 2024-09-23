import os
import csv
import pandas as pd
import torch
from torch.utils.data import DataLoader
from dgllife.utils import AttentiveFPAtomFeaturizer, AttentiveFPBondFeaturizer, EarlyStopping
from rdkit import Chem
from model import MyGNN
from process import mydataset
from utils import collate_molgraphs, split_molecules_by_color, filter_mol, draw_mol, parse_args

if torch.cuda.is_available():
    print('use GPU')
    torch.cuda.manual_seed(1000)
    device = torch.device('cuda:0')
else:
    print('use CPU')
    torch.manual_seed(2000)
    device = torch.device('cpu')

args = parse_args()
pred_path = args.pred_path
model_path = args.model_path
frag_path = args.frag_path
image_path = args.image_path

filter_mol(pred_path, 'tmp_1.csv')

node_featurizer = AttentiveFPAtomFeaturizer(atom_data_field='feat')
edge_featurizer = AttentiveFPBondFeaturizer(bond_data_field='feat')

model_ = MyGNN(node_input_size=node_featurizer.feat_size(), edge_input_size=edge_featurizer.feat_size(),
               num_gnn_layers=4, num_update_times=2,
               graph_size=182, hidden_size1=39, hidden_size2=8,
               output_size=1).to(device)

stopper_ = EarlyStopping(patience=30, filename=model_path, metric='roc_auc_score')
stopper_.load_checkpoint(model_)

print('-----------pred dataset---------------')
pred_data = mydataset(csv_name='tmp_1.csv', graph_name='./pred.bin')
pred_loader = DataLoader(dataset=pred_data, batch_size=1,
                         collate_fn=collate_molgraphs)

os.remove('tmp_1.csv')

labels_all = []
outputs_all = []

model_.eval()
for i, batch_data in enumerate(pred_loader):
    smiles, bg, labels, masks = batch_data

    bg = bg.to(device)
    node_feats = bg.ndata['feat'].to(device)
    edge_feats = bg.edata['feat'].to(device)

    out, att = model_(bg, node_feats, edge_feats, probability1=0.361, probability2=0.139)

    _, pes_frags = split_molecules_by_color(smiles, bg, att, threshold=0.5)
    pes_frag_list = [Chem.MolToSmiles(mol) for mol in pes_frags]

    pes_frag_split = pes_frag_list[0].split('.')

    # with open('./number.csv', mode='a', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow([len(pes_frag_split)])

    filtered_pes_frag = [frags_p for frags_p in pes_frag_split if len(frags_p) >= 3]

    with open('tmp_2.csv', 'a') as file:
        for pes_frag in filtered_pes_frag:
            file.write(pes_frag + '\n')

df = pd.read_csv('tmp_2.csv', header=None)
df.columns = ['smiles']

df['smiles_lower'] = df['smiles'].str.lower()
df = df[~df['smiles_lower'].duplicated(keep='first')]
df = df.drop('smiles_lower', axis=1)
df.reset_index(drop=True, inplace=True)

df.to_csv('tmp_3.csv', index=False)
os.remove('tmp_2.csv')

filter_mol('tmp_3.csv', frag_path)
os.remove('tmp_3.csv')

draw_mol(frag_path, image_path)
os.remove('./pred.bin')
