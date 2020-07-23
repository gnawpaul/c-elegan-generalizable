import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import generate_edge_index
from torch_geometric.nn import MessagePassing, GatedGraphConv, GINConv, GlobalAttention, GATConv, SAGEConv
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_scatter import scatter_add
from torch_geometric.utils import softmax

### Graph level classification-Classification on raw dataset

class MLPClassifier(torch.nn.Module):

    def __init__(self, n_hid, states, nodes, cat=True):
        '''Simple MLP Classifier'''
        super(MLPClassifier, self).__init__()
        if cat:
            n_in = nodes*n_hid
        else:
            n_in = n_hid
            
        self.MLP_enc  = MLP(2, n_hid, n_hid, bn=True, run_stats=False)
        self.MLP_out  = MLP(nodes*n_hid, n_hid, n_hid)
        self.fc = nn.Linear(n_hid, states)
        self.nodes = nodes
        self.n_hid = n_hid

    def forward(self, x, class_args):
        # x.shape: [batch_size*latent_nodes, timesteps, dims]
        nodes = self.nodes
        batch_size = int(x.size(0)/nodes)
        timesteps = x.size(1)
        dims = x.size(2)
        x = x.view(batch_size, nodes, timesteps, dims).transpose(1,2).contiguous()
        x = x.reshape(batch_size*timesteps, nodes,dims)
        x = self.MLP_enc(x)
        # x.shape: [batch_size, timesteps, nodes, n_hid]
        x = self.MLP_out(x.reshape(batch_size*timesteps, nodes*self.n_hid))
        x = F.softmax(self.fc(x),dim=-1)
        return x, torch.zeros(1)

class Graph_Attention(nn.Module):

    def __init__(self, n_hid, states, nodes, cat=True, dynamic_graph=False):
        super(Graph_Attention, self).__init__()
        layers = 1

        self.mlp_enc = MLP(2, n_hid, n_hid, bn=True, run_stats=False)
        self.gconv = GNNConv_Weighted(n_hid, n_hid, n_hid)
        self.att = MLP_Attention(n_hid, cat=True)
        self.aggr_mlp= MLP(n_hid,n_hid,1, bn=True)

        if cat==True:
            n_in = n_hid*nodes*(layers+1)
        else:
            n_in = n_hid*(layers+1)
        self.mlp_out = MLP(n_in, n_hid, n_hid, bn=True)
        self.fc      = nn.Linear(n_hid, states)

        self.n_hid  = n_hid
        self.nodes  = nodes
        self.layers = layers
        self.cat    = cat
        self.dynamic_graph = dynamic_graph

    def forward(self, x, class_args):
        # x.shape: [batch_size*nodes, timesteps, n_hid]

        assert 'timesteps'        in class_args
        assert 'batch_edge_index' in class_args

        timesteps  = class_args['timesteps']
        edge_index = class_args['batch_edge_index']
        batch      = class_args['batch']
        nodes      = self.nodes
        n_hid      = self.n_hid
        dims       = x.size(-1)
        batch_size = int(x.size(0)/nodes)
        row, col   = edge_index
        
        #  Embed Node Features
        x = x.view(batch_size, nodes, timesteps, dims).transpose(1,2)
        x = x.reshape(batch_size*timesteps*nodes,dims)
        # x.shape: [batch_size*nodes*timesteps, dims]
        feat = self.mlp_enc(x,channel=False)
        hid = []
        edge_attr_list = []
        hid.append(feat)
        edge_attr, batch_edge_attr = self.att(x, edge_index, batch_size, nodes, timesteps, dynamic_graph=self.dynamic_graph )
        edge_attr_list.append(edge_attr)
        feat = self.gconv(feat, edge_index, batch_edge_attr)
        hid.append(feat)
        edge_attr = torch.stack(edge_attr_list, dim=-1)
        hid = torch.stack(hid, dim=-1)
        # hid.shape: [batch_size*timesteps*nodes, n_hid, layers]
        hid = hid.reshape(batch_size*timesteps*nodes, n_hid, (2))
        # hid.shape[batch_size, nodes, timesteps, n_hid]
        if self.cat:
            hid = hid.reshape(batch_size*timesteps,2*nodes*(n_hid))
        else:
            hid = hid.reshape(batch_size*timesteps,nodes, n_hid*(2))
            hid = torch.sum(hid,dim=1)
        # [batch_size*timesteps, hid*layers]

        hid = self.fc(self.mlp_out(hid,channel=False))
        hid = F.softmax(hid, dim=-1)
        # hid.shape: [batch_size*timesteps, states, 1]
        return hid, edge_attr


class MLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, n_in, n_hid, n_out, do_prob=0., bn=False, run_stats=False):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        #self.bn_mid  = nn.BatchNorm1d(n_hid)
        self.bn_out  = nn.BatchNorm1d(n_out,track_running_stats=run_stats)
        self.bn = bn
        self.dropout_prob = do_prob
        self.n_out, self.n_hid = n_out, n_hid
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # batch norm where x:[N, C, L]
    def batch_norm_channel(self, x):
        n_out = self.n_out
        size = x.shape
        x = x.transpose(-1,-2)
        x = self.bn_out(x.view(-1,n_out, x.size(-1)))
        x = x.transpose(-1,-2).view(size)
        return x

    # batch norm where x: [N, C]
    def batch_norm(self,x):
        n_out = self.n_out
        size = x.shape
        x    = self.bn_out(x.view(-1,n_out))
        x    = x.view(size)
        return x

    def forward(self, inputs, channel=False):
        n_out, n_hid = self.n_out, self.n_hid
        # Input shape: [num_sims, num_things, num_features]
        x = F.elu(self.fc1(inputs))
        x = F.elu(self.fc2(x))

        if self.bn is False:
            return x
        
        if channel:
            x = self.batch_norm_channel(x)
        else:
            x = self.batch_norm(x)

        return x

class MLP_Attention(torch.nn.Module):

    def __init__(self, n_hid, cat=True):
        '''MLP for calculating attention'''
        super(MLP_Attention, self).__init__()
        self.mlp_enc = MLP(2, n_hid, n_hid)
        if cat:
            self.mlp = MLP(2*n_hid, n_hid, n_hid, bn=True)
        else:
            self.mlp = MLP(n_hid, n_hid, n_hid, bn=True)
        self.fc  = nn.Linear(n_hid,2)
        self.n_hid = n_hid
        self.cat   = cat

    def forward(self, x, edge_index, batch_size, nodes, timesteps, dynamic_graph=False):
        # x.shape: [batch_size*timesteps*nodes, dims]
        row, col = edge_index
        n_hid = self.n_hid
        x = self.mlp_enc(x)
        if self.cat:
            edge_attr = torch.cat((x[row,:],x[col,:]),dim=-1)
        else:
            edge_attr = x[row,:] + x[col,:]
        edge_attr = self.mlp(edge_attr)
        edge_attr = edge_attr.reshape(batch_size, timesteps, nodes**2, n_hid)
        if not dynamic_graph:
            edge_attr = torch.mean(edge_attr,dim=(0,1))
            edge_attr = torch.squeeze(F.softmax(self.fc(edge_attr), dim=-1)[:,1])
            batch_edge_attr = edge_attr.repeat(batch_size*timesteps)
        else:
            edge_attr = torch.squeeze(F.softmax(self.fc(edge_attr), dim=-1)[:,:,:,1])
            batch_edge_attr = edge_attr
        return edge_attr, batch_edge_attr

# This module is originally based of the GIN in pytorch geometric; however, after our modifications,
#   it's essentially a generic GNN. Some residual code may still exist.
class GNNConv_Weighted(MessagePassing):

    def __init__(self, n_in, n_hid, n_out):
        super(GNNConv_Weighted, self).__init__(aggr='add')
        self.mlp  = MLP(n_in, n_hid, n_hid,
                        bn=True, run_stats=False)
        self.fc_out   = nn.Linear(n_hid, n_out)


    def forward(self, x, edge_index, edge_attr, **kwargs):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        out = self.mlp(self.propagate(edge_index, x=x, edge_attr=edge_attr), channel=False)
        return self.fc_out(out)

    def message(self, x_j, edge_attr):
        msg = torch.einsum('ab,a->ab', x_j, edge_attr.view(-1))
        return msg

class Node_MLP(nn.Module):
    def __init__(self, dims, n_hid, nodes):
        super(Node_MLP, self).__init__()
        self.mlp    = MLP(dims, n_hid, n_hid,  b_norm=True)
        self.fc_out = nn.Linear(n_hid, dims)
        self.dims   = dims

    def forward(self, inputs, edge_index, edge_attr, **kwargs):
        x = self.mlp(inputs)
        x = self.fc_out(x)
        return x

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)
            
def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)