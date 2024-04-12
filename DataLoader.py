import scipy as sp
from scipy.sparse import coo_matrix
import numpy as np
import networkx as nx
import pandas as pd
import torch
import scanpy as sc
from scipy.sparse import *

class load_data(object):
    def __init__(self,**kwargs):
        allowed_arg = ['grn_path','feature_path']
        for kwarg in kwargs.keys():
            assert kwarg in allowed_arg,'Invalid kwargs: '+kwarg+'; file_path is needed'
        self.grn_path = kwargs.get('grn_path')
        self.feature_path = kwargs.get('feature_path')
        # storing network data
        self.data = {
            'features': torch.nn.Identity(),
            'adj': torch.nn.Identity(),
            'adj_orig': torch.nn.Identity()
        }
        # storing a dict of coord with a gene
        self.gene_dict = dict()
        self.graph = nx.DiGraph()
    #construct networkx graph object
    def read_grn_csv(self):
        grn_csv = pd.read_csv(self.grn_path, sep=",")
        grn_csv = grn_csv[['source', 'target', 'coef_mean']]
        graph = nx.DiGraph()
        gene_set = set(grn_csv['source'].tolist() + grn_csv['target'].tolist())
        for i in gene_set:
            graph.add_node(i)
        for i in range(grn_csv.shape[0]):
            graph.add_edge(grn_csv['source'].iloc[i],
                           grn_csv['target'].iloc[i],
                           weight=grn_csv['coef_mean'].iloc[i])
        adj_matrix = nx.adjacency_matrix(graph)
        self.graph = graph
        self.data['adj_orig'] = torch.tensor(adj_matrix.toarray()).to(torch.float32)
    # read gene features data
    def read_node_features(self):
        # adata = sc.read_h5ad(self.feature_path)
        # nodes = list(GRN_obj.gene_dict.values())
        # x = adata[:,nodes].X
        # x = x.T
        # x = torch.tensor(x).to(torch.float32)
        filename = self.feature_path
        df = pd.read_csv(filename)
        x = torch.tensor(df.iloc[:,1:].to_numpy()).float()
        self.data['features'] = x

    # make a dict to save the coords of each gene symbol
    def coord_gene_dict(self):
        num_nodes = len(self.graph.nodes())
        coords = range(num_nodes)
        gene_symbol = self.graph.nodes()
        dic = dict(zip(coords, gene_symbol))
        self.gene_dict = dic

    def laplacian_norm(self):
        adj = coo_matrix(self.data.get('adj_orig'))
        adj_ = adj + sp.sparse.eye(adj.shape[0])  # add self features for each nodes
        # Out-degree normalization of adj (see section 3.3.1 of paper)
        degree_mat_inv_sqrt = sp.sparse.diags(np.power(np.array(adj_.sum(1)), -1).flatten())
        adj_normalized = degree_mat_inv_sqrt.dot(adj_)
        adj_normalized = torch.tensor(adj_normalized.toarray()).to(torch.float32)
        self.data['adj'] = adj_normalized


def adj2list(adj):
    m = adj
    if not isspmatrix_coo(m):
        m = coo_matrix(m)
    idx = np.vstack((m.row, m.col))
    return torch.tensor(idx,dtype=torch.int64),torch.tensor(m.data),m.shape
