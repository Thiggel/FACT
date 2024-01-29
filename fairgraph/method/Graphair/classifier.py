import torch
import torch.nn as nn

from fairgraph.method.Graphair.GCN import GCN_Body

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim) -> None:
        super(Classifier,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1)
        )
    
    def forward(self,h):
        return self.model(h)
    
    def reset_parameters(self) -> None:
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

class GCNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0, n_layer=2) -> None:
        super(Classifier,self).__init__()

        self.gcn = GCN_Body(in_feats=input_dim, n_hidden=hidden_dim, out_feats=hidden_dim, dropout=dropout, n_layer=n_layer)

        self.model = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, adj, features):
        h = self.gcn(adj, features)
        return self.model(h)
    
    def reset_parameters(self) -> None:
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
        for layer in self.gcn.layers:
            layer.init_params()
