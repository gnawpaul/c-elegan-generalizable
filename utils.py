import os
import sys
import time
import pytz
import torch
import shutil
import datetime

import pickle as pk
import numpy  as np
import matplotlib.pyplot as plt

from pytz import timezone
import torch.nn.functional as F
from scipy.sparse import coo_matrix
from torch_geometric.data import Data, DataLoader

def print_train_valid(epoch, losses, end='', file=sys.stdout):
    print('Epoch: {:04d} \n'.format(epoch), file=file, end=end)
    currentDT = datetime.datetime.now(tz=pytz.utc).astimezone(timezone('US/Pacific'))
    print('time:',currentDT.strftime("%Y-%m-%d: %I:%M:%S %p"),'\n',file=file, end=end)
    print(losses ,'\n', file=file, end=end)
    file.flush()
    return

# Return files where model will be saved.
def save_file_log(folder_path, overwrite):
    abs_path = os.path.abspath('')
    path = os.path.join(abs_path, folder_path)
    folder_name = os.path.split(folder_path)[1]
    # If folder already exists, delete it if <overwrite> is <True>.
    if os.path.exists(path) and overwrite:
        try:
            shutil.rmtree(path)
            print("'%s'overwritten" %path)
        except:
            print ("Unexpected error:", sys.exc_info()[0])
    if not os.path.exists(path):
        # Make folders and files.
        os.makedirs(path, exist_ok=True)

    class_init     = os.path.join(path, 'class_init.pt')
    class_best     = os.path.join(path, 'class_best.pt')
    log_file_class = os.path.join(path, 'log.txt')
    buf_file_class = os.path.join(path, 'buffer.pkl')
    with open(log_file_class, 'w') as log:
        currentDT = datetime.datetime.now(tz=pytz.utc).astimezone(timezone('US/Pacific'))
        print('{}'.format(folder_name),file=log)
        print(currentDT.strftime("%Y-%m-%d: %I:%M:%S %p")+"\n",file=log)
        print("Save folder: {}".format(folder_path), file=log)
    return class_init, class_best, log_file_class, buf_file_class

# Generates edge index. Returns the adjacency matrix in COO format.
def generate_edge_index(nodes1, nodes2):
    edge_index = torch.LongTensor(np.array(np.where(np.ones((nodes1,nodes2))), dtype='int'))
    return edge_index

# Generate shuffled data. Returns data divided into 10 portions where data is randomized. Returned
#   data is first element of datasets where other elements/datasets are used to find common
#   uniquely identified neurons.
def generate_data_lists(datasets):

    assert all([data in ['Kato', 'n2', 'npr1', 'Kato-Original'] for data in datasets])

    train_valid_size = 8
    all_data  = get_data(datasets)
    tot_worms = get_num_worms(all_data[0])
    worms     = list(range(tot_worms))

    # Get data for each worm.
    train_valid_worm = {}
    test_worm  = {}
    for worm_index in worms:
        neuron_ind = unique_neuron_indices(all_data[0], all_data, worm_index)
        raw_data = all_data[0]
        tr      = get_trace(raw_data,worm_index)[neuron_ind]
        diff_tr = get_diff_trace(raw_data,worm_index)[neuron_ind]
        states = get_states(raw_data,worm_index)
        assert tr.shape == diff_tr.shape
        data = []
        for trace, dtrace in zip(tr, diff_tr):
            data.append(np.array((normalize_data(trace), normalize_data(dtrace), states)))
        data = np.array(data)
        num_sets = int(np.floor(data.shape[2]/8))

        # Calculate the number of extra timesteps.
        excessTimesteps = int(data.shape[2] % (8))
        # Delete excess timesteps and reshape such that the first dimension is number of batches.
        batched_data = np.delete(data, np.arange(excessTimesteps),axis=2).transpose(2,0,1)
        batched_data = batched_data.reshape(-1,8,batched_data.shape[1],batched_data.shape[2])
        np.random.shuffle(batched_data)
        # Split numpy array along batch axis, combine as array, and squeeze.
        batched_data = np.squeeze(np.array(np.split(batched_data, num_sets)))
        batched_data = batched_data.transpose(0,2,1,3)
        # numpy arrays of shape [samples, num_neurons, timesteps, 3]
        batched_data = torch.FloatTensor(batched_data)
        data = []
        for element in batched_data:
            data.append(Data(x=element[:,:,0:2],state=torch.unsqueeze(element[0,:,2:3],dim=0).long()))
        test_size = int(len(data)/10)
        train_valid_worm.update({worm_index+1: data[:-3*test_size]})
        test_worm.update({worm_index+1: data[-3*test_size:]})

    return train_valid_worm, test_worm, int(len(neuron_ind))

# Continuous data for evaluation
def generate_data_eval(train_dataset, eval_dataset):


    assert all([dataset in ['Kato', 'n2', 'npr1','Kato-Original'] for dataset in eval_dataset])
    assert train_dataset in ['Kato', 'n2', 'npr1','Kato-Original']
    eval_data, train_data = get_data(eval_dataset), get_data([train_dataset])[0]
    worms = np.arange(get_num_worms(eval_data[0]))

    data_loader = []
    for worm_index in worms:
        neuron_ind = unique_neuron_indices(eval_data[0], eval_data, worm_index)
        raw_data = eval_data[0]
        tr      = get_trace(raw_data,worm_index)[neuron_ind]
        diff_tr = get_diff_trace(raw_data,worm_index)[neuron_ind]
        states = get_states(raw_data,worm_index)
        assert tr.shape == diff_tr.shape
        data = []
        for trace, dtrace in zip(tr, diff_tr):
            data.append(np.array((normalize_data(trace), normalize_data(dtrace), states)))
        data = np.array(data).transpose(0,2,1)
        data = torch.FloatTensor(data)
        x      = data[:,:,0:2]
        states = torch.squeeze(data[0,:,2:3]).long()
        data_loader.append([x, states])
    return data_loader

def get_data(files):
    path = os.path.abspath('')
    data = []
    for folder in files:
        data_path = os.path.join(path,'Data',folder)
        with open(data_path,'rb') as file:
            data.append(np.squeeze(pk.load(file)))
    return data

# return neuron ids (Shape: [num_neurons])
def get_ids(data, worm_index):
    return np.squeeze(data[worm_index][3])

# return calcium traces (Shape: [num_neurons, timesteps])
def get_trace(data, worm_index):
    return data[worm_index][1].transpose(1,0)

# return derivative of calcium traces (Shape: [num_neurons, timesteps])
def get_diff_trace(data, worm_index):
    return data[worm_index][2].transpose(1,0)

# return states (Shape: [timesteps])
def get_states(data, worm_index):
    return np.squeeze(data[worm_index][7])

# return time in seconds (Shape: [timesteps])
def get_time(data, worm_index):
    return np.squeeze(data[worm_index][4])

# return name of state. <input>: <int> (from 1 to 8)
def get_state_name(data, state):
    return np.concatenate(data[0][8].tolist()[0]).tolist()[state-1]

# Return number:<int> of worms in a dataset.
def get_num_worms(data):
    return int(np.size(data, 0))

#### Functions that are used to process unique neurons

# Return a set of uniquely identified neurons.
def unique_id(data, worm_index):
    indices = unique_indices(data, worm_index)
    ids=[]
    for element in get_ids(data, worm_index)[indices].tolist():
        ids.append(np.squeeze(element[0].tolist()).tolist())
    return set(ids)

# Return list of indices of uniquely identified neurons of specified worm.
def unique_indices(data, worm_index):
    indices=[]
    # If the id vector has only 1 element (uniquely identified), append the index to the returned list.
    for i, element in enumerate(get_ids(data, worm_index).tolist()):
        if element.size is 1:
            indices.append(i)
    return indices

# Return list of alphabetically sorted list of names of uniquely identified neurons common across all 5 worms.
def get_unique_neurons(data, worm_indices_list):
    # data is a list of data. worm_index is a list of worm's requested where each element is a list
    #   of requested worms.
    neurons = []
    for dataset, worm_indices in zip(data, worm_indices_list):
        neurons.append(set.intersection(*[unique_id(dataset, worm_index-1) for worm_index in worm_indices]))  
    unique_neurons = list(set.intersection(*neurons))
    unique_neurons.sort()
    return unique_neurons

# Return list of indices for uniquely identified neurons (across all worms) of the specified worm.
def unique_neuron_indices(data, all_data, worm_index):
    # all_data includes all datasets that we want unique indices. data is just the dataset we are interested in.
    # Get indices for uniquely identified neurons.
    indices = unique_indices(data, worm_index)
    ids={}
    # Create a dict: {neuron_name: neuron_number} for specified worm.
    for i, element in enumerate(get_ids(data, worm_index)[indices].tolist()):
        ids.update({np.squeeze(element[0].tolist()).tolist():indices[i]})
    # Get alphabetically sorted list of common uniquely identified neurons across all 5 worms.
    unique_neurons = get_unique_neurons(all_data, [np.arange(get_num_worms(element)) for element in all_data])
    # Get list of neuron indices corresponding to basis defined by <unique_neurons>.
    neuron_index = []
    for element in unique_neurons:
        neuron_index.append(ids[element])
    return neuron_index

# Return normalized 1D data with range (-1,1).
def normalize_data(data):
    return (data-data.min())*2/(data.max()-data.min())-1
