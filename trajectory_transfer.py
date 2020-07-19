import os
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

from utils import *
from trajectory_modules import *
from torch_geometric.data import DataLoader
from itertools import combinations
from torch.optim import lr_scheduler

def train(epoch, best_val_loss, log, train_loader, eval_data):

    edge_encoder.train(), predictor.train()
    loss_data = {}
    loss_fn    = nn.MSELoss()
    mse_train = []
    for data_loader in train_loader:
        for batch_idx, batch in enumerate(data_loader):

            # Zero grad
            optimizer.zero_grad()

            x = batch.x.to(device) # x.shape: [nodes*batch_size, timesteps, dim]
            training_args.update({'epoch':epoch})
            edges  = edge_encoder(x, edge_index, dynamic_graph)
            # edges.shape: [(batch_size, timesteps), edges]
            output = predictor(x, edge_index, edges, training_args)
            # output.shape: [batch_size*nodes, timesteps-1, dims]

            ### Loss
            target  = x[:,1:,:]
            loss    = loss_fn(output, target)
            # Backprop and step optimizer
            loss.backward()
            optimizer.step()
            mse_train.append(loss.item())
    loss_data.update({'mse_train':mse_train})

    # Evaluation
    edge_encoder.eval(), predictor.eval()
    with torch.no_grad():
        loss_fn  = nn.MSELoss(reduction='none')
        mse_tot_eval = []
        k = 0
        for strain in eval_data:
            mse_strain_eval  = []
            for worm in strain:
                x, states = worm[0].to(device), worm[1].to(device)
                edges  = edge_encoder(x, edge_index, dynamic_graph)
                output = predictor(x, edge_index, edges, inference_args)
                target  = x[:,1:,:]
                loss    = loss_fn(output, target)
                mse_strain_eval.append(loss)
                mse_tot_eval.append(torch.mean(loss))
            if k == 0:
                loss_data.update({'n2_mse':mse_strain_eval})
            elif k==1:
                loss_data.update({'npr1_mse':mse_strain_eval})
            elif k==2:
                loss_data.update({'Kato_mse':mse_strain_eval})
            k+=1
    mean_mse_loss = torch.mean(torch.FloatTensor([torch.mean(worm) for worm in mse_tot_eval])).item()
    print_train_valid(epoch, mean_mse_loss, file=log)
    #Save model and losses for best validation
    if mean_mse_loss < best_val_loss:
        pk.dump(loss_data, open(buf_file, "wb"))
    scheduler.step()
    return mean_mse_loss, loss_data

# This code replicates the results in our paper on only the Kato dataset for transfer learning.
np.random.seed(42)
torch.manual_seed(42)
t = time.time()

dim  = 2       # <int>: Number of dimensions of the features(2 for our dataset).
cat  = True    # <bool>: <True> to use concatenation for aggregation.
cuda = True    # <bool>: <True> to use CUDA.
n_hidden = 128 # <int>: Number of hidden units/dimensions for MLP.
lr = 1e-3      # <float>: Learning rate.
gamma = .25    # <float>: Learning rate decay rate.
lr_decay = 200 # <int>: Number of iterations to decay learning rate by gamma.
dynamic_graph = False # <bool>: <True> to allow weights to change each timestep. Otherwise, weight remains
                     #   same for the whole temporal graph.
save_folder = "logs/Node Trajectory/Eval_Other_Worms_Static_128"
device = torch.device('cuda' if (torch.cuda.is_available() and cuda) else 'cpu')

# Get <list> of <arrays> where each <array> corresponds to data from a worm in Kato dataset.
train_list, test_list, nodes = generate_data_lists(['Kato', 'n2', 'npr1'])
# Combine test and train list since we validate on the other worms.
data = {}
for key in train_list.keys():
    data.update({key: train_list[key]})
for key in test_list.keys():
    data[key].extend(test_list[key])

# Generate edge index and calculate number of worms.
edge_index = generate_edge_index(nodes, nodes).to(device)
num_worms = np.array(list(train_list.keys())).max()

output = {}
for model_type in ['GNN_Att','MLP']:
    print('Starting {} model.'.format(model_type))
    i = 1 
    while i != num_worms+1:
        print('Training worms: {}'.format(i))
        folder_path = os.path.join(save_folder, 'Number of training worms: {} | Model type: {}'.format(i, model_type))
        _, _, log_file, buf_file = save_file_log(folder_path, False)
        training_worms = list(combinations(list(train_list.keys()), i))

        for perm in training_worms:
            # Evaluate on unseen worms.
            print(perm)
            eval_worms   = [worm-1 for worm in list(train_list.keys()) if worm not in perm]
            kato_eval_data = generate_data_eval('Kato',['Kato','npr1','n2'])

            # Model
            eval_data    = [generate_data_eval('Kato',['n2','npr1','Kato']), generate_data_eval('Kato',['npr1','n2','Kato']), [kato_eval_data[i] for i in eval_worms]]
            train_loader = [DataLoader(data[element],batch_size=600) for element in perm]

            edge_encoder = MLP_Attention(n_hidden, nodes).to(device)
            predictor    = Predictor(n_hidden, nodes, edge_encoder, model_type).to(device)
            training_args = {'dynamic_graph':dynamic_graph, 'scheduling':'scheduled_sampling',
                    'epoch':0,'last_epoch':275} 
            inference_args = {'dynamic_graph':dynamic_graph, 'scheduling':'teacher', 'prediction_steps':16}

            best_val_loss = np.inf
            best_epoch = 0
            with open(log_file, 'a') as log:
                optimizer = optim.Adam(list(edge_encoder.parameters())+list(predictor.parameters()),lr=lr)
                scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay,gamma=gamma)
                for epoch in range(300):
                    val_loss, loss_data = train(epoch, best_val_loss, log, train_loader, eval_data)
                    if val_loss < best_val_loss:
                        best_val_loss  = val_loss
                        best_loss_data = loss_data
                        best_loss_data.update({'epoch':epoch})
                output.update({str(model_type)+' | '+str(perm): best_loss_data})
        i += 1
pk.dump( output, open(os.path.join(save_folder,'output.pk'), "wb"))
print(time.time()-t)