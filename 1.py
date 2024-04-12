import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.datasets import Planetoid
import networkx as nx
import scanpy as sc
from torch.nn import Module
from torch import Tensor
from torch_geometric.utils import negative_sampling

"""
    layers
"""


# ----Encoder
class GraphSAGE(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 hidden,
                 z_dim,
                 dropout,
                 act=F.relu,
                 aggr='mean'):
        super(GraphSAGE, self).__init__()
        self.sage1 = SAGEConv(in_channels=input_dim,
                              out_channels=hidden,
                              aggr=aggr,
                              normalize=False,
                              root_weight=True,
                              project=False,
                              bias=True)
        self.sage2_mu = SAGEConv(in_channels=hidden,
                                 out_channels=z_dim,
                                 aggr=aggr,
                                 normalize=False,
                                 root_weight=True,
                                 project=False,
                                 bias=True)
        self.sage2_std = SAGEConv(in_channels=hidden,
                                  out_channels=z_dim,
                                  aggr=aggr,
                                  normalize=False,
                                  root_weight=True,
                                  project=False,
                                  bias=True)
        self.act = act
        self.dropout = dropout

    def forward(self, features: Tensor, edges: Tensor):
        features = self.sage1(features, edges)
        features = self.act(features)
        features = F.dropout(features, p=self.dropout)
        mu = self.sage2_mu(features, edges)
        # mu = self.act(mu)
        std = self.sage2_std(features, edges)
        # std = self.act(std)
        return mu, std


class GAT(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 hidden,
                 z_dim,
                 dropout,
                 head,
                 act=F.relu):
        super(GAT, self).__init__()
        self.gat1 = GATConv(in_channels=input_dim,
                            out_channels=hidden,
                            heads=head,
                            concat=True,
                            # If set to False, the multi-head attentions are averaged instead of concatenated
                            bias=True)

        self.gat2_mu = GATConv(in_channels=hidden * head,
                               out_channels=z_dim,
                               heads=1,
                               concat=True,
                               # If set to False, the multi-head attentions are averaged instead of concatenated
                               bias=True)

        self.gat2_std = GATConv(in_channels=hidden * head,
                                out_channels=z_dim,
                                heads=1,
                                concat=True,
                                # If set to False, the multi-head attentions are averaged instead of concatenated
                                bias=True)
        self.act = act
        self.dropout = dropout

    def forward(self, features: Tensor, edges: Tensor):
        features = self.gat1(features, edges)
        features = self.act(features)
        features = F.dropout(features, p=self.dropout)
        mu = self.gat2_mu(features, edges)
        # mu = self.act(mu)
        std = self.gat2_std(features, edges)
        # std = self.act(std)
        return mu, std


# ----Decoder
class InnerProductDecoder(torch.nn.Module):
    """
        simplest decoder for VGAE: InnerProduct-Decoder
    """

    def forward(self, z: Tensor):
        outputs = torch.matmul(z, z.t())
        # outputs = torch.relu(outputs)
        return outputs


class sourcetargetdecoder(torch.nn.Module):
    """
        Source Target Decoder
    """

    def forward(self, z: Tensor):
        _z_dim = z.shape[1]
        _dim = torch.tensor(_z_dim / 2, dtype=int)
        # source embedding for each nodes
        input_source = z[:, 0:_dim]
        # targget embedding for each nodes
        input_target = z[:, _dim:_z_dim]
        # source*target to reconstruct adjacency matrix
        outputs = torch.matmul(input_source, input_target.T)
        # outputs = torch.relu(outputs)
        return outputs


class GravityDecoder(torch.nn.Module):
    """
        gravity inspired decoder
    """

    def __init__(self, epsilon, lamda):
        super(GravityDecoder, self).__init__()
        self.epsilon = epsilon
        self.lamda = lamda

    def forward(self, z: Tensor):
        _z_dim = z.shape[1]
        # Get mass
        mass = z[:, (_z_dim - 1):_z_dim]
        # mass = mass.transpose(0,1)
        # get z(z_dim-1)
        r = z[:, 0:(_z_dim - 1)]
        # calculate distence
        dist = torch.cdist(r, r, p=2) + self.epsilon
        # Gravity-Inspired decoding
        outputs = mass - self.lamda * torch.log(dist)
        # outputs = F.relu(outputs)
        return outputs


"""
    model
"""


class VGAE(torch.nn.Module):
    def __init__(self,
                 encoder: Module,
                 decoder: Module):
        super(VGAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        # self.features = node_features
        # self.edges = edges
        # self.adj = adj
        self.mu = torch.tensor([0.1])
        self.logstd = torch.tensor([0.1])
        self.z = torch.tensor([0.1])

    def reparametrize(self, mu: Tensor, logstd: Tensor) -> Tensor:
        return mu + torch.randn_like(logstd) * torch.exp(logstd)

    def _encode(self, features, edges) -> Tensor:
        self.mu, self.logstd = self.encoder(features, edges)
        z = self.reparametrize(mu=self.mu, logstd=self.logstd)
        self.z = z
        return z

    def forward(self,
                features,
                edges) -> Tensor:
        z = self._encode(features, edges)
        output = self.decoder(z=z)
        return output

    def recon_loss(self,
                   x: Tensor,
                   adj: Tensor,
                   pos_edge_index: Tensor,
                   neg_edge_index: Tensor = None) -> Tensor:
        adj = adj
        my_pos_loss = F.mse_loss(x[pos_edge_index[0], pos_edge_index[1]], adj[pos_edge_index[0], pos_edge_index[1]])

        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, adj.shape[0])
        my_neg_loss = F.mse_loss(x[neg_edge_index[0], neg_edge_index[1]], adj[neg_edge_index[0], neg_edge_index[1]])

        return my_pos_loss + my_neg_loss

    def kl_loss(self, mu: Tensor = None, logstd: Tensor = None) -> Tensor:
        mu = self.mu if mu is None else mu
        logstd = self.logstd if logstd is None else logstd
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu ** 2 - logstd.exp() ** 2, dim=1))