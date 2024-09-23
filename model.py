import torch.nn as nn
import torch.nn.functional as F
from dgllife.model.gnn.attentivefp import AttentiveFPGNN
from dgllife.model.readout.attentivefp_readout import AttentiveFPReadout


class MyGNN(nn.Module):
    def __init__(self, node_input_size, edge_input_size,
                 num_gnn_layers, num_update_times, graph_size,
                 hidden_size1, hidden_size2, output_size):
        super(MyGNN, self).__init__()
        self.afp = AttentiveFPGNN(node_feat_size=node_input_size,
                                  edge_feat_size=edge_input_size,
                                  num_layers=num_gnn_layers,
                                  graph_feat_size=graph_size)
        self.readout = AttentiveFPReadout(feat_size=graph_size,
                                          num_timesteps=num_update_times)
        self.fc1 = nn.Linear(graph_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, node_feats, edge_feats, probability1, probability2, get_node_weight=True):
        node_feats = self.afp(g, node_feats, edge_feats)
        if get_node_weight:
            graph_feats, node_weights = self.readout(g, node_feats, get_node_weight)
            x1 = F.relu(self.fc1(graph_feats))
            x1 = F.dropout(x1, p=probability1, training=self.training)
            x2 = F.relu(self.fc2(x1))
            x2 = F.dropout(x2, p=probability2, training=self.training)
            out = self.sigmoid(self.fc3(x2))
            return out, node_weights
