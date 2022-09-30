import torch
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops

from torch_geometric.nn import GraphConv, TopKPooling, GatedGraphConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F

class SAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(SAGEConv, self).__init__(aggr='max')  # "Max" aggregation.
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.act = torch.nn.ReLU()
        self.update_lin = torch.nn.Linear(in_channels + out_channels, in_channels, bias=False)
        self.update_act = torch.nn.ReLU()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j):
        # x_j has shape [E, in_channels]
        x_j = self.lin(x_j)
        x_j = self.act(x_j)

        return x_j

    def update(self, aggr_out, x):
        # aggr_out has shape [N, out_channels]
        new_embedding = torch.cat([aggr_out, x], dim=1)

        new_embedding = self.update_lin(new_embedding)
        new_embedding = self.update_act(new_embedding)

        return new_embedding




class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # num_items = df.item_id.max() + 1
        num_items = 30
        embed_dim = 32

        self.conv1 = SAGEConv(embed_dim, embed_dim)
        self.pool1 = TopKPooling(embed_dim, ratio=0.8)
        self.conv2 = SAGEConv(embed_dim, embed_dim)
        self.pool2 = TopKPooling(embed_dim, ratio=0.8)
        self.conv3 = SAGEConv(embed_dim, embed_dim)
        self.pool3 = TopKPooling(embed_dim, ratio=0.8)
        self.item_embedding = torch.nn.Embedding(num_embeddings= num_items + 1, embedding_dim=embed_dim)

        self.lin1 = torch.nn.Linear(int(embed_dim*2), embed_dim)
        self.lin2 = torch.nn.Linear(embed_dim, int(embed_dim//2))
        self.lin3 = torch.nn.Linear(int(embed_dim//2), 1)
        # self.bn1 = torch.nn.BatchNorm1d(128)
        # self.bn2 = torch.nn.BatchNorm1d(64)
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.item_embedding(x)

        x = x.squeeze(1)
        x = self.conv1(x, edge_index)
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.conv2(x, edge_index)
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.conv3(x, edge_index)
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        embed = x1 + x2 + x3

        x = self.lin1(embed)
        x = self.act1(x)
        x = self.lin2(x)
        x = self.act2(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = torch.sigmoid(self.lin3(x)).squeeze(1)

        return x, embed.squeeze()