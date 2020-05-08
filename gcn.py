import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool, JumpingKnowledge


class GCN(torch.nn.Module):
    def __init__(self, args):
        self.args = args

        hidden = 16
        num_layers = 3

        super(GCN, self).__init__()
        self.conv1 = GCNConv(args.n_node_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, 1)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, weights = data.x, data.edge_index, data.weight
        weights = weights.float()


        x = F.relu(self.conv1(x, edge_index, weights))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, weights))
        #x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        c = self.lin2(x)
        choice = F.softmax(c, dim=0).view(self.args.n_nodes)

        #print(choice)
        #input()



        v = global_mean_pool(x, torch.zeros(data.num_nodes, dtype=torch.long).to(self.args.device))
        value = self.lin2(v)

        return choice, value.squeeze()

    def __repr__(self):
        return self.__class__.__name__


class GCNWithJK(torch.nn.Module):
    def __init__(self, args, mode='cat'):
        super(GCNWithJK, self).__init__()

        self.args = args
        hidden = 16
        num_layers = 3

        self.conv1 = GCNConv(args.n_node_features, hidden)
        self.conv3 = GCNConv(hidden, 1)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.jump = JumpingKnowledge(mode)
        if mode == 'cat':
            self.lin1 = Linear(num_layers * hidden, hidden)
        else:
            self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, 1)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, weights = data.x, data.edge_index, data.weight
        weights = weights.float()


        x = F.relu(self.conv1(x, edge_index, weights))
        x = F.dropout(x)
        xs = [x]
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, weights))
            x = F.dropout(x)
            xs += [x]
        x = self.jump(xs)
        #x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x)
        c = self.conv3(x, edge_index, weights)
        choice = F.softmax(c, dim=0).view(self.args.n_nodes)


        v = global_mean_pool(x, torch.zeros(data.num_nodes, dtype=torch.long).to(self.args.device))
        value = self.lin2(v)

        return choice, value

    def __repr__(self):
        return self.__class__.__name__
