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

def train(epoch, best_val_loss, log, train_loader, test_loader):

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

    # Validation
    edge_encoder.eval(), predictor.eval()
    with torch.no_grad():
        loss_fn   = nn.MSELoss(reduction='none')
        test_loss = []
        k = 0
        for data_loader in test_loader:
            for batch_idx, batch in enumerate(data_loader):
                x = batch.x.to(device)
                edges  = edge_encoder(x, edge_index, dynamic_graph)
                output = predictor(x, edge_index, edges, validation_args)
                target  = x[:,1:,:]
                loss    = loss_fn(output, target)
                test_loss.append(torch.mean(loss))
        mean_mse_loss = torch.mean(torch.FloatTensor(test_loss)).item()
        print_train_valid(epoch, mean_mse_loss, file=log)
        if mean_mse_loss < best_val_loss:
            pk.dump(loss_data, open(buf_file, "wb"))
        scheduler.step(mean_mse_loss)
    return mean_mse_loss


def evaluate(eval_data, loss_data):

    # Evaluation
    edge_encoder.eval(), predictor.eval()
    with torch.no_grad():
        #loss_fn  = nn.MSELoss(reduction='none')
        loss_fn = torch.nn.L1Loss(reduction='none')
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
            if k == 0:
                loss_data.update({'n2_mse':mse_strain_eval})
            elif k==1:
                loss_data.update({'npr1_mse':mse_strain_eval})
            elif k==2:
                loss_data.update({'Kato_mse':mse_strain_eval})
            k+=1
    return loss_data


# This code replicates the results in our paper on only the Kato dataset for transfer learning.
np.random.seed(0)
torch.manual_seed(0)
t = time.time()

dim  = 2       # <int>: Number of dimensions of the features(2 for our dataset).
cat  = True    # <bool>: <True> to use concatenation for aggregation.
cuda = True    # <bool>: <True> to use CUDA.
n_hidden = 256 # <int>: Number of hidden units/dimensions for MLP.
dynamic_graph = True # <bool>: <True> to allow weights to change each timestep. Otherwise, weight remains
                     #   same for the whole temporal graph.
save_folder = "logs/Node Trajectory/Eval_256_Tanh"
device = torch.device('cuda' if (torch.cuda.is_available() and cuda) else 'cpu')

# Get <list> of <arrays> where each <array> corresponds to data from a worm in Kato dataset.
train_list, test_list, nodes = generate_data_lists(['Kato', 'n2', 'npr1'])

# Generate edge index and calculate number of worms.
edge_index = generate_edge_index(nodes, nodes).to(device)
num_worms = np.array(list(train_list.keys())).max()

output = {}
for model_type in ['GNN_Att','MLP']:
    if model_type == 'GNN_Att':
        lr = .5e-3
        gamma = .5
        lr_decay = 500
    else:
        lr = 1e-3
        gamma = .5
        lr_decay = 500
    print('Starting {} model.'.format(model_type))
    i = 1 
    while i!=num_worms+1:
        print('Training worms: {}'.format(i))
        folder_path = os.path.join(save_folder, 'Number of training worms: {} | Model type: {}'.format(i, model_type))
        _, _, log_file, buf_file = save_file_log(folder_path, False)
        training_worms = list(combinations(list(train_list.keys()), i))

        for perm in training_worms:
            # Evaluate on unseen worms.
            print(perm)
            save_path = os.path.join(save_folder, 'Number of training worms: {} | Model type: {}'.format(i, model_type), str(perm))
            eval_worms   = [worm-1 for worm in list(train_list.keys()) if worm not in perm]
            kato_eval_data = generate_data_eval('Kato',['Kato','npr1','n2'])

            # Model
            eval_data    = [generate_data_eval('Kato',['n2','npr1','Kato']), generate_data_eval('Kato',['npr1','n2','Kato']), [kato_eval_data[i] for i in eval_worms]]
            train_loader = [DataLoader(train_list[element],batch_size=600) for element in perm]
            test_loader  = [DataLoader(test_list[element],batch_size=600) for element in perm]

            edge_encoder = MLP_Attention(n_hidden, nodes).to(device)
            predictor    = Predictor(n_hidden, nodes, edge_encoder, model_type).to(device)
            training_args = {'dynamic_graph':dynamic_graph, 'scheduling':'scheduled_sampling',
                    'epoch':0,'last_epoch':275} 
            validation_args = {'dynamic_graph':dynamic_graph, 'scheduling':'teacher', 'prediction_steps':8}
            inference_args = {'dynamic_graph':dynamic_graph, 'scheduling':'teacher', 'prediction_steps':16}

            best_val_loss = np.inf
            best_epoch = 0
            best_loss_data = {}
            with open(log_file, 'a') as log:
                optimizer = optim.Adam(list(edge_encoder.parameters())+list(predictor.parameters()),lr=lr)
                scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50)
                #scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay,gamma=gamma)
                for epoch in range(600):
                    val_loss = train(epoch, best_val_loss, log, train_loader, test_loader)
                    if val_loss < best_val_loss:
                        best_val_loss  = val_loss
                        torch.save({'edge_encoder':edge_encoder.state_dict(),
                                    'predictor':predictor.state_dict()}, save_path)
                        best_loss_data.update({'epoch':epoch})
                        best_loss_data.update({'val_loss':val_loss})
                best_model = torch.load(save_path)
                edge_encoder.load_state_dict(best_model['edge_encoder'])
                predictor.load_state_dict(best_model['predictor'])
                best_lost_data = evaluate(eval_data, best_loss_data)
                output.update({str(model_type)+' | '+str(perm): best_loss_data})
        i += 1
pk.dump( output, open(os.path.join(save_folder,'output.pk'), "wb"))
print(time.time()-t)