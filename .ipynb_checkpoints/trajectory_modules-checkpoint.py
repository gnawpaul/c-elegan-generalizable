import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops

class Predictor(nn.Module):
    """Trajectory prediction module."""

    def __init__(self, n_hid, nodes, encoder, decoder_type='MLP'):
        super(Predictor, self).__init__()
        assert decoder_type in ['MLP','GNN_Att']
        self.n_hid = n_hid
        self.nodes = nodes
        self.decoder_type = decoder_type
        self.encoder = encoder

        if decoder_type == 'MLP':
            self.gnn  = NRI_MLP(2, n_hid, nodes)
        elif decoder_type == 'GNN_Att':
            self.gnn  = GNNConv_Weighted(2, n_hid, 2)

    def single_step_forward(self, inputs, edge_index, edges, hidden=None):
        # inputs.shape: [batch_size, num_timesteps/every nth time step, nodes, dims]
        # edges.shape: [(batch_size, timesteps/nth time step if dynamic_graph), edges]
        batch_size    = self.batch_size
        n_hid = self.n_hid
        nodes = self.nodes
        timesteps = inputs.size(1)
        size_1 = batch_size*timesteps*nodes
        gnn_args = {'batch_size': batch_size,'nodes':nodes,'n_hid':n_hid,'timesteps':timesteps}
        x = inputs.reshape(size_1, 2)
        # Generate edge index for batch.
        buffer = []
        for i in range(batch_size*timesteps):
            buffer.append((i*nodes)+edge_index)
        batch_edge_index = torch.cat(buffer,dim=1)
        # If dynamic graph, reshape edge weights. Otherwise, duplicate edge weights.
        assert edges.dim() in (3,1)
        if edges.dim() == 3:
            edges = edges.reshape(-1)
        else:
            edges = edges.repeat(batch_size*timesteps)
        assert edges.size(0) == batch_size*timesteps*nodes**2
            
        x = self.gnn(x, batch_edge_index, edges, gnn_args=gnn_args)
        # x.shape: [batch_size*timesteps*nodes, dims]

        x = x.view(batch_size, timesteps, nodes, 2)

        # Predict position/velocity difference
        return inputs + x, hidden

    def forward(self, inputs, edge_index, edges, decoder_args):
        # inputs.shape:[batch_size*nodes, timesteps, dims]

        assert 'scheduling' in decoder_args
        assert decoder_args['scheduling'] in ['scheduled_sampling', 'burn_in', 'teacher']
        assert 'dynamic_graph'      in decoder_args
        scheduling    = decoder_args['scheduling']
        dynamic_graph = decoder_args['dynamic_graph']
        # If schedule sampling, get parameters for schedule. Otherwise, get number of prediction steps.
        if scheduling == 'scheduled_sampling':
            assert 'epoch'      in decoder_args
            assert 'last_epoch' in decoder_args
            epoch, last_epoch = decoder_args['epoch'], decoder_args['last_epoch']
        elif scheduling == 'teacher':
            assert 'prediction_steps' in decoder_args
            pred_steps = decoder_args['prediction_steps']
        else:
            assert 'burn_in_steps' in decoder_args
            burn_in = decoder_args['burn_in_steps']
        n_hid, nodes = self.n_hid, self.nodes
        dims  = inputs.size(-1)
        batch_size = int(inputs.size(0)/nodes)
        timesteps  = inputs.size(1)
        self.dynamic_graph = dynamic_graph
        self.batch_size = batch_size

        #edge_index, _ = remove_self_loops(edge_index)
        row, col      = edge_index

        inputs = inputs.view(batch_size, nodes, timesteps, inputs.size(-1)).transpose(1,2).contiguous()
        # inputs.shape: [batch_size, timesteps, nodes, dims] 

        preds = []
        hid   = None

        if scheduling == 'scheduled_sampling':
            for step in range(timesteps-1):
                if not dynamic_graph:
                            edges_pred = edges
                if epoch < last_epoch:
                    # Use scheduled sampling algorithm with linear decay if training.
                    prob = 1-epoch/last_epoch
                    m = torch.distributions.bernoulli.Bernoulli(torch.tensor([prob])).sample()
                    if m or step == 0:
                        last_pred = torch.unsqueeze(inputs[:,step,:,:],dim=1)
                        if dynamic_graph:
                            edges_pred = torch.unsqueeze(edges[:,step,:],dim=1)
                    elif dynamic_graph:
                        x_in = last_pred.transpose(1,2).reshape(last_pred.size(0)*last_pred.size(2),last_pred.size(1),last_pred.size(3))
                        edges_pred = self.encoder(x_in,edge_index, {'dynamic_graph':True,'one_hot':False})
                else:
                    if step == 0:
                        last_pred = torch.unsqueeze(inputs[:,step,:,:],dim=1)
                    if dynamic_graph:
                        x_in = last_pred.transpose(1,2).reshape(last_pred.size(0)*last_pred.size(2),last_pred.size(1),last_pred.size(3))
                        edges_pred = self.encoder(x_in,edge_index, {'dynamic_graph':True,'one_hot':False})
                if edges_pred.dim()==2:
                    edges_pred = torch.unsqueeze(edges_pred, dim=1)
                last_pred, hid = self.single_step_forward(last_pred, edge_index, edges_pred, hid)
                preds.append(last_pred)
            pred_all = torch.squeeze(torch.stack(preds, dim=-2))
        elif scheduling == 'teacher':
            assert (pred_steps <= timesteps)
            # Only take n-th timesteps as starting points (n: pred_steps)
            last_pred  = inputs[:, 0::pred_steps, :, :]
            if not dynamic_graph:
                edges_pred = edges
            else:
                edges_pred = edges[:, 0::pred_steps,:]

            # Run n prediction steps
            for step in range(0, pred_steps):
                last_pred, hid = self.single_step_forward(last_pred, edge_index, edges_pred, hid)
                if dynamic_graph:
                    x_in = last_pred.transpose(1,2).reshape(last_pred.size(0)*last_pred.size(2),last_pred.size(1),last_pred.size(3))
                    edges_pred = self.encoder(x_in, edge_index, {'dynamic_graph':True,'one_hot':False})
                    if edges_pred.dim() == 2:
                            edges_pred = torch.unsqueeze(edges_pred, dim=1)
                preds.append(last_pred)

            sizes = [preds[0].size(0), preds[0].size(1) * pred_steps,
                     preds[0].size(2), preds[0].size(3)]
            output = torch.zeros(sizes).to(inputs.device)
            # Re-assemble correct timeline
            for i in range(len(preds)):
                output[:, i::pred_steps, :, :] = preds[i]
            pred_all = output[:, :(timesteps - 1), :, :]
            pred_all = pred_all.transpose(1, 2).contiguous()
            # pred_all.shape: [batch_size, nodes, timesteps-1, dims]
        else:
            assert burn_in < timesteps
            assert burn_in > 0
            for step in range(timesteps-1):
                if not dynamic_graph:
                    edges_pred_step = edges
                else:
                    edges_pred_step = torch.unsqueeze(edges[:,step,:],dim=1)
                if step+1 <= burn_in:
                    last_pred = torch.unsqueeze(inputs[:,step,:,:],dim=1)
                last_pred, hid = self.single_step_forward(last_pred, edge_index, edges_pred_step, hid)
                preds.append(last_pred)
            pred_all = torch.squeeze(torch.stack(preds, dim=-2))
        return torch.reshape(pred_all, (batch_size*nodes, timesteps-1,dims))
        # pred_all.shape: [samples*nodes, timesteps, dim]

# While the overall setup for this module is the same as that in the classification module,
#  this module is different because of we iterate over timesteps instead of the more efficient
#  batch edge index message passing. We also run this module separately from the decoder; however,
#  this was done to unclutter the predictor module. This module can be easily combine with the
#  predictor module as seen in the publication.
class MLP_Attention(nn.Module):
    def __init__(self, n_hid, nodes):
        super(MLP_Attention, self).__init__()
        self.nodes = nodes
        self.n_hid = n_hid

        self.mlp1 = MLP(2, n_hid, n_hid, bn=True)
        self.mlp2 = MLP(2*n_hid , n_hid, n_hid, bn=True)
        self.fc_out = nn.Linear(n_hid, 2)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def forward(self, inputs, edge_index, dynamic_graph=True):

        # Input shape: [batch_size*nodes, timesteps, dims]
        
        n_hid = self.n_hid
        nodes = self.nodes
        edges = nodes ** 2 -15
        batch_size = int(inputs.size(0)/nodes)
        timesteps  = inputs.size(1)
        row, col      = edge_index

        x = inputs.view(batch_size, nodes, timesteps, inputs.size(2))
        # x.shape: [batch_size, nodes, timesteps, dims] 
        x = x.transpose(1,2).contiguous()
        # x.shape: [batch_size, timesteps, nodes, dims] 

        # Embed x
        x = self.mlp1(x, channel=False)  # 2-layer ELU net per node
        # x.shape: [num_sims, timesteps, nodes, n_hid]
        # Set edge features as the concat of nodes features connected by the edge.
        x = torch.cat((x[:,:,row,:],x[:,:,col,:]),dim=-1)
        # x.shape: [batch_size, timesteps, edges, n_hid]

        if not dynamic_graph:
            x = torch.mean(x, dim =(0,1))
        x = self.mlp2(x, channel=False)
        # x.shape: [batch_size, timesteps, edges, n_hid]

        x = F.softmax(self.fc_out(x), dim=-1)
        if not dynamic_graph:
            x = torch.squeeze(x[:,1])
        else:
            x = torch.squeeze(x[:,:,:,1])
            if batch_size == 1:
                x = torch.unsqueeze(x, dim=0)
        # x.shape: [(batch_size, timesteps), edges]
        return x


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


class NRI_MLP(nn.Module):
    def __init__(self, dims, n_hid, nodes):
        super(NRI_MLP, self).__init__()
        self.mlp    = MLP(nodes*dims, n_hid, n_hid,  bn=True)
        self.fc_out = nn.Linear(n_hid, nodes*dims)
        self.dims   = dims

    def forward(self, inputs, edge_index, edge_attr, **kwargs):
        args = kwargs['gnn_args']
        x = inputs.view(args['batch_size']*args['timesteps'],args['nodes']*self.dims)
        x = self.mlp(x)
        x = self.fc_out(x)
        x = x.view(args['batch_size']*args['timesteps']*args['nodes'],self.dims)
        return x

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