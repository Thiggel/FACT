import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
from .GCN import GCN_Body, GAT_Body


class aug_module(torch.nn.Module):
    def __init__(
        self,
        features: torch.tensor,
        n_hidden: int = 64,
        temperature: float = 1,
        device: str = 'cpu',
        dropout: float = 0.1,
        nlayer: int = 1,
        mlpx_dropout: float = 0.1,
        edge_perturbation=True,
        node_feature_masking=True,
        use_graph_attention: bool = False,
        normalize: bool = True,
    ) -> None:
        super(aug_module, self).__init__()
        self.device = device

        self.g_encoder = GCN_Body(
            in_feats=features.shape[1],
            n_hidden=n_hidden,
            out_feats=n_hidden,
            dropout=dropout,
            nlayer=nlayer
        ) if not use_graph_attention else GAT_Body(
            in_feats=features.shape[1],
            n_hidden=n_hidden,
            out_feats=n_hidden,
            dropout=dropout,
            nlayer=nlayer
        )

        self.use_graph_attention = use_graph_attention

        self.Aaug = MLPA(in_feats=n_hidden, dim_h=n_hidden, dim_z=features.shape[1])
        self.Xaug = MLPX(in_feats=n_hidden, n_hidden=n_hidden, out_feats=features.shape[1], dropout=mlpx_dropout)
        self.edge_perturbation = edge_perturbation
        self.node_feature_masking = node_feature_masking
        
        self.temperature = temperature
        self.normalize = normalize

    def forward(self, adj, x, alpha=0.5, adj_orig=None):
        h = self.g_encoder(adj, x)

        adj_logits = None

        if self.edge_perturbation:
            adj_logits = self.Aaug(h)
            edge_probs = torch.sigmoid(adj_logits)

            if adj_orig is not None:
                edge_probs = alpha*edge_probs + (1-alpha)*adj_orig

            adj_sampled = pyro.distributions.RelaxedBernoulliStraightThrough(
                temperature=self.temperature,
                probs=edge_probs
            ).rsample()
            # making adj_sampled symmetric
            adj_sampled = adj_sampled.triu(1)
            adj_sampled = adj_sampled + adj_sampled.T

            if not self.use_graph_attention and self.normalize:
                adj_sampled = self.normalize_adj(adj_sampled)
        else:
            adj_sampled = adj
            adj_logits = None

        # Node feature masking
        if self.node_feature_masking:
            mask_probs = torch.sigmoid(self.Xaug(h))
            mask = pyro.distributions.RelaxedBernoulliStraightThrough(
                temperature=self.temperature,
                probs=mask_probs
            ).rsample()
            x_new = x * mask
        else:
            x_new = x

        return adj_sampled, x_new, adj_logits
    
    def normalize_adj(self,adj):
        adj.fill_diagonal_(1)
        # normalize adj with A = D^{-1/2} @ A @ D^{-1/2}
        D_norm = torch.diag(torch.pow(adj.sum(1), -0.5)).to(self.device)
        adj = D_norm @ adj @ D_norm
        return adj

class MLPA(torch.nn.Module):

    def __init__(self, in_feats, dim_h, dim_z):
        super(MLPA, self).__init__()
        
        self.gcn_mean = torch.nn.Sequential(
                torch.nn.Linear(in_feats, dim_h),
                torch.nn.ReLU(),
                torch.nn.Linear(dim_h, dim_z)
                )

    def forward(self, hidden):
        # GCN encoder
        Z = self.gcn_mean(hidden)
        # inner product decoder
        adj_logits = Z @ Z.T
        return adj_logits

class MLPX(torch.nn.Module):

    def __init__(self, in_feats, n_hidden, out_feats, dropout):
        super(MLPX, self).__init__()

        self.dropout = nn.Dropout(dropout)

        self.fc1 = torch.nn.Linear(in_feats, n_hidden)
        self.fc2 = torch.nn.Linear(n_hidden, out_feats)
    
    def projection(self, z):
        z = F.relu(self.fc1(z))
        return self.fc2(z)

    def forward(self, h):
        return self.projection(h)

