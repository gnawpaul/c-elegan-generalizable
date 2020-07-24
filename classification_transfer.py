import os
import torch

import numpy as np
import torch.optim as optim
import torch.nn.functional as F

from utils import *
from class_modules import *
from torch_geometric.data import DataLoader
from itertools import combinations
from torch.optim import lr_scheduler

def train(epoch, best_val_loss, log, train_loader, test_loader):

    nll_train  = []
    cor_train  = []
    class_data = {}
    loss_fn = nn.NLLLoss(ignore_index=num_states-1)

    class_model.train()
    loss_data = {}
    for data_loader in train_loader:
        for batch_idx, batch in enumerate(data_loader):

            # Zero grad
            optimizer.zero_grad()

            x = batch.x.to(device) # x.shape: [nodes*batch_size, timesteps, dim]

            states    = batch.state.to(device)
            batch_num = batch.batch
            batch_size= batch_num.max().item()+1
            batch_buffer=torch.arange(batch_size*8).unsqueeze(-1).expand((-1,nodes)).reshape(-1).to(device)
            buffer = []
            for i in range((batch.batch.max()+1)*8):
                buffer.append((i*nodes)+edge_index)
            batch_edge_index = torch.cat(buffer,dim=1)
            class_args.update({'batch':batch_buffer, 'timesteps':8,
                               'batch_edge_index':batch_edge_index})

            output,_ = class_model(x, class_args)
            # output.shape: [batch_size*timesteps, classes]

            ### Loss
            target  = torch.squeeze(states-1).view(-1)
            output= torch.cat((output,torch.zeros(output.size(0),1).to(device)),dim=-1)
            loss    = loss_fn(torch.log(output.view(-1,num_states)), target)
            # Backprop and step optimizer
            loss.backward()
            optimizer.step()

            # Record errors
            nll_train.append(loss.data.item())
            _, indices  = torch.max(output,dim=-1)
            comp_corr = ((indices.view(-1) == target)|(target==num_states-1)).sum()
            comp_ign  = (target == num_states-1).sum()
            cor = ((comp_corr-comp_ign).float()/(target!=num_states-1).sum()).item()
            cor_train.append(cor)

    loss_data.update({'nll_train':nll_train, 'correct_train':cor_train})

    # Validation
    class_model.eval()
    with torch.no_grad():
        nll_val, cor_val = [], []
        for data_loader in test_loader:
            for batch_idx, batch in enumerate(data_loader):
                x = batch.x.to(device) # x.shape: [nodes*batch_size, timesteps, dim]

                states    = batch.state.to(device)
                batch_num = batch.batch
                batch_size= batch_num.max().item()+1
                batch_buffer = torch.arange(batch_size*8).unsqueeze(-1).expand((-1,nodes)).reshape(-1).to(device)
                buffer = []
                for i in range((batch.batch.max()+1)*8):
                    buffer.append((i*nodes)+edge_index)
                batch_edge_index = torch.cat(buffer,dim=1)
                class_args.update({'batch':batch_buffer, 'timesteps':8,
                                   'batch_edge_index':batch_edge_index})

                output, _ = class_model(x, class_args)
                # output.shape: [batch_size*timesteps, classes]

                ### Loss
                target  = torch.squeeze(states-1).view(-1)
                output= torch.cat((output,torch.zeros(output.size(0),1).to(device)),dim=-1)
                loss    = loss_fn(torch.log(output.view(-1,num_states)), target)

                # Record errors
                nll_val.append(loss.data.item())
                _, indices  = torch.max(output,dim=-1)
                comp_corr = ((indices.view(-1) == target)|(target==num_states-1)).sum()
                comp_ign  = (target == num_states-1).sum()
                cor = ((comp_corr-comp_ign).float()/(target!=num_states-1).sum()).item()
                cor_val.append(cor)
        scheduler.step(np.mean(nll_val))
    loss_data.update({'nll_val':nll_val, 'correct_val':cor_val})
    return np.mean(nll_val), loss_data
    
def evaluate(eval_data, loss_data):
    class_model.eval()
    loss_fn = nn.NLLLoss(ignore_index=num_states-1)
    with torch.no_grad():
        nll_tot_eval = []
        k = 0
        for strain in eval_data:
            nll_strain_eval  = []
            strain_corr_eval = []
            for worm in strain:
                x, states = worm[0].to(device), worm[1].to(device)
                timesteps_eval = x.size(1)
                batch_buffer = torch.arange(timesteps_eval).unsqueeze(-1).expand((-1,nodes)).reshape(-1).to(device)
                buffer = []
                for i in range(timesteps_eval):
                    buffer.append((i*nodes)+edge_index)
                batch_edge_index = torch.cat(buffer,dim=1)
                class_args.update({'batch':batch_buffer, 'timesteps':timesteps_eval,
                               'batch_edge_index':batch_edge_index})
                output, att = class_model(x, class_args)
                ### Loss
                target  = torch.squeeze(states-1).view(-1)
                output= torch.cat((output,torch.zeros(output.size(0),1).to(device)),dim=-1)
                loss    = loss_fn(torch.log(output.view(-1,num_states)), target)
                _, indices  = torch.max(output,dim=-1)
                nll_eval = loss.data.item()
                comp_corr = ((indices.view(-1) == target)|(target==num_states-1)).sum()
                comp_ign  = (target == num_states-1).sum()
                cor = ((comp_corr-comp_ign).float()/(target!=num_states-1).sum()).item()
                nll_tot_eval.append(nll_eval)
                nll_strain_eval.append(nll_eval)
                strain_corr_eval.append(cor)
                per_results = {"num_ignored":comp_ign,"predicted_states": indices, "ground_truth":target}
            if k == 0:
                loss_data.update({'n2_nll':nll_strain_eval, 'n2_corr':strain_corr_eval, 'results':per_results})
            elif k==1:
                loss_data.update({'npr1_nll':nll_strain_eval, 'npr1_corr':strain_corr_eval, 'results':per_results})
            elif k==2:
                loss_data.update({'Kato_nll':nll_strain_eval, 'Kato_corr':strain_corr_eval, 'results':per_results})
            k+=1
    return loss_data

# This code replicates the results in our paper on only the Kato dataset for transfer learning.
np.random.seed(0)
torch.manual_seed(0)
t = time.time()

dim  = 2    # <int>: Number of dimensions of the features(2 for our dataset).
cat  = True # <bool>: Whether or not to use concatenation for aggregation.
cuda = True
n_hidden = 12
dynamic_graph = False # <bool>: <True> to allow changing weights each timestep.
save_folder = "logs/Graph Classification/Eval_5_States_12"
device = torch.device('cuda' if (torch.cuda.is_available() and cuda) else 'cpu')

# Experiments purely on Kato dataset (15 common neurons).
num_states = 5
train_list, test_list, nodes = generate_data_lists(['Kato', 'n2', 'npr1'])

edge_index = generate_edge_index(nodes, nodes).to(device)
num_worms = np.array(list(train_list.keys())).max()

output = {}
for model_type in ['GNN_Att','MLP']:
    print('Starting {} model.'.format(model_type))
    i = 1
    if model_type == 'GNN_Att':
        lr = 1e-3
        gamma = .25
        lr_decay = 600
    else:
        lr = 1e-3
        gamma = .25
        lr_decay = 600
    while i != num_worms+1:
        print('Training worms: {}'.format(i))
        folder_path = os.path.join(save_folder, 'Number of training worms: {} | Model type: {}'.format(i, model_type))
        class_init, class_best, log_file, buf_file = save_file_log(folder_path, False)
        training_worms = list(combinations(list(train_list.keys()), i))

        for perm in training_worms:
            # Evaluate on unseen worms.
            print(perm)
            save_path = os.path.join(save_folder, 'Number of training worms: {} | Model type: {}'.format(i, model_type), str(perm))
            eval_worms   = [worm-1 for worm in list(train_list.keys()) if worm not in perm]
            kato_eval_data = generate_data_eval('Kato',['Kato','npr1','n2'])
                
            eval_data    = [generate_data_eval('Kato',['n2','npr1','Kato']), generate_data_eval('Kato',['npr1','n2','Kato']), [kato_eval_data[i] for i in eval_worms]]
            train_loader = [DataLoader(train_list[element],batch_size=600) for element in perm]
            test_loader  = [DataLoader(test_list[element],batch_size=600) for element in perm]
            if model_type == 'MLP':
                class_model = MLPClassifier(n_hidden, num_states, nodes, cat=cat).to(device)
            elif model_type == 'GNN_Att':
                class_model = Graph_Attention(n_hidden, num_states, nodes,cat=cat, dynamic_graph=dynamic_graph).to(device)
            class_args = {}
            best_val_loss = np.inf
            best_epoch = 0
            with open(log_file, 'a') as log:
                optimizer = optim.Adam(list(class_model.parameters()),lr=lr)
                scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50)
                for epoch in range(800):
                    val_loss, loss_data = train(epoch, best_val_loss, log, train_loader, test_loader)
                    if val_loss < best_val_loss:
                        best_val_loss  = val_loss
                        best_loss_data = loss_data
                        best_loss_data.update({'epoch':epoch})
                        torch.save(class_model.state_dict(), save_path)
                class_model.load_state_dict(torch.load(save_path))
                best_lost_data = evaluate(eval_data, best_loss_data)
                output.update({str(model_type)+' | '+str(perm): best_loss_data})
        i += 1
pk.dump( output, open(os.path.join(save_folder,'output.pk'), "wb"))
print(time.time()-t)
