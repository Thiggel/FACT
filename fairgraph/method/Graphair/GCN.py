import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch_geometric.nn.models import GAT
from scipy.sparse import csr_matrix


class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, activation=F.relu, dropout=None, bias=True):
        super(GCNLayer, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.activation = activation
        if bias:
            self.b = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.b = None
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        self.init_params()

    def init_params(self):
        if self.W is not None:
            init.xavier_uniform_(self.W)
        if self.b is not None:
            init.zeros_(self.b)

    def forward(self, adj, h):
        if self.dropout:
            h = self.dropout(h)
        x = h @ self.W

        x = adj @ x
        if self.b is not None:
            x = x + self.b
        if self.activation:
            x = self.activation(x)
        return x


class GCN_Body(nn.Module):
    def __init__(self, in_feats, n_hidden, out_feats, dropout, nlayer):
        super(GCN_Body, self).__init__()
        self.layers = nn.ModuleList()

        # input layer
        self.layers.append(GCNLayer(in_feats, n_hidden))
        # hidden layers
        for i in range(nlayer - 2):
            self.layers.append(GCNLayer(n_hidden, n_hidden))
        # output layer
        self.layers.append(GCNLayer(n_hidden, out_feats))

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, x):
        h = x
        cnt = 0
        for layer in self.layers:
            if self.dropout and cnt != 0:
                h = self.dropout(h)
            cnt += 1
            h = (layer(g, h))
        return h


class GAT_Body(nn.Module):
    def __init__(
        self,
        in_feats: int,
        n_hidden: int,
        out_feats: int,
        dropout: float,
        nlayer: int,
    ) -> None:
        super(GAT_Body, self).__init__()

        self.gat = GAT(
            in_channels=in_feats,
            hidden_channels=n_hidden,
            out_channels=out_feats,
            num_layers=nlayer,
            dropout=dropout,
        )

    def forward(self, adj, x):
        if not adj.is_sparse:
            adj = adj.to_sparse_coo()

        print(adj)
        return self.gat(x, adj.coalesce())


class GCN(nn.Module):
    def __init__(
        self,
        in_feats: int,
        n_hidden: int,
        out_feats: int,
        nclass: int,
        dropout: float = 0.2,
        nlayer: int = 2
    ):
        super(GCN, self).__init__()

        self.create_body(in_feats, n_hidden, out_feats, dropout, nlayer)

        self.fc = nn.Sequential(
            nn.Linear(out_feats, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, nclass),
        )

    def create_body(self, in_feats, n_hidden, out_feats, dropout, nlayer):
        self.body = GCN_Body(in_feats, n_hidden, out_feats, dropout, nlayer)

    def forward(self, g, x):
        h = self.body(g, x)
        x = self.fc(h)
        return x, h


class GAT_Model(GCN):
    def create_body(self, in_feats, n_hidden, out_feats, dropout, nlayer):
        self.body = GAT_Body(in_feats, n_hidden, out_feats, dropout, nlayer)
