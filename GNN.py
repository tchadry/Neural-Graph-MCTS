import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

import torch_geometric
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader

# understand what use_gdc is, what num_nodes_features and n_nodes are
# args we need: n_nodes_features, n_nodes


class GNN(nn.Module):
    def __init__(self, args):
        # what to do with use_gdc
        
        super(GNN, self).__init__()
        self.args=args
        d=args.n_node_features
        num_nodes = args.n_nodes
        
        self.conv1 = GCNConv(d,  16,normalize=not args.use_gdc)
        self.conv2 = GCNConv(16,  16,normalize=not args.use_gdc)
        self.conv3 = GCNConv(16, 1,normalize=not args.use_gdc)
        # then, we have to return a results for pi and v, respectively
        self.activ1 = nn.Linear(16, num_nodes)
        self.activ2 = nn.Linear(16, 1)
        # define parameters
        self.reg_params = self.conv1.parameters()
        self.non_reg_params = self.conv2.parameters()

    def forward(self, data):
        #print("analyzing data")
        print(data)

        x, edges = data.x, data.edge_index

        x = self.conv1(x, edges)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.conv2(x, edges)
        x = F.relu(x)

        c = self.conv3(x, edges)
        #choice = torch.masked_select(c.squeeze(), choices)
        choice = F.softmax(c, dim=0).view(self.args.n_nodes)

        v = global_mean_pool(x, torch.zeros(data.num_nodes, dtype=torch.long))
        value = self.activ2(v)

        print("value: ", value.squeeze())

        return choice, value.squeeze()

