import numpy as np
import scipy.sparse as sp
import torch
import matplotlib.pyplot as plt
from glob import glob
import os
import re
from ast import literal_eval

def scipysp_to_pytorchsp(sp_mx):
    """ converts scipy sparse matrix to pytorch sparse matrix """
    if not sp.isspmatrix_coo(sp_mx):
        sp_mx = sp_mx.tocoo()
    coords = np.vstack((sp_mx.row, sp_mx.col)).transpose()
    values = sp_mx.data
    shape = sp_mx.shape
    pyt_sp_mx = torch.sparse.FloatTensor(torch.LongTensor(coords.T),
                                         torch.FloatTensor(values),
                                         torch.Size(shape))
    return pyt_sp_mx

def accuracy(output, labels):
    output = output.squeeze()
    preds = (output>0).type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def fair_metric(output,idx, labels, sens):
    val_y = labels[idx].cpu().numpy()
    idx_s0 = sens.cpu().numpy()[idx.cpu().numpy()]==0
    idx_s1 = sens.cpu().numpy()[idx.cpu().numpy()]==1

    idx_s0_y1 = np.bitwise_and(idx_s0,val_y==1)
    idx_s1_y1 = np.bitwise_and(idx_s1,val_y==1)

    pred_y = (output[idx].squeeze()>0).type_as(labels).cpu().numpy()
    parity = abs(sum(pred_y[idx_s0])/sum(idx_s0)-sum(pred_y[idx_s1])/sum(idx_s1))
    equality = abs(sum(pred_y[idx_s0_y1])/sum(idx_s0_y1)-sum(pred_y[idx_s1_y1])/sum(idx_s1_y1))

    return parity,equality

def set_device():
    """
    Function for setting the device.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    try:
        if torch.backends.mps.is_available():
            device = torch.device('mps')
    except:
        device = torch.device('cpu')
    return device

def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    except:
        pass

def find_pareto_front(results, metric1, metric2):
    """In this method, the first metric is maximized, the second is minimized!!"""
    pareto_front = []
    for i in range(len(results)):
        is_dominated = False
        for j in range(len(results)):
            if i == j: continue
            if results[i][metric1]['mean'] <= results[j][metric1]['mean'] and \
                results[i][metric2]['mean'] >= results[j][metric2]['mean']:
                is_dominated = True

        if not is_dominated:
            pareto_front.append(results[i])  
            
    return pareto_front

def plot_pareto(results, fairness_metric, title, show_all, filepath=None):
    """This only properly works if metric2 is acc"""
    # Create a new figure
    plt.figure()

    # find the pareto front
    pareto_front = find_pareto_front(results, "acc", fairness_metric)

    # collect all the points (also ones not in the front)
    all_points = np.zeros((len(results),2))
    for i in range(len(results)):
        all_points[i][0] = results[i]["acc"]['mean']
        all_points[i][1] = results[i][fairness_metric]['mean']
    
    # create the xs and ys for the plot
    arr = np.zeros((len(pareto_front), 2))
    for i, r in enumerate(pareto_front):
        arr[i][0] = r[fairness_metric]['mean']
        arr[i][1] = r["acc"]['mean']
    arr = np.sort(arr, axis=0)

    # create the plot
    if show_all:
        plt.scatter(all_points[:, 1], all_points[:, 0], color='red')

    plt.plot(arr[:, 0], arr[:, 1], color='blue')
    plt.scatter(arr[:, 0], arr[:, 1], color='blue')
    plt.xlabel(fairness_metric.upper())
    plt.ylabel("accuracy")
    plt.title(title)
    
    if filepath is not None:
        plt.savefig(filepath)
    else:
        plt.show()

def get_grid_search_result_files(dir):
    """Returns a list of paths to all the output files in the given directory."""
    return glob(os.path.join(dir, "output-a*.txt"))

def get_grid_search_results_from_dir(dir):
    """Returns a tuple: (all_avg_results, finshed_hparams), where results is a list of dicts containing average results,
    and finished_hparams is a list of lists of hyperparameter values for which the experiment has been completed."""

    paths = get_grid_search_result_files(dir)

    all_avg_results = [], finished_hparams = []
    for path in paths:
        a, _, g, l = [float(param) for param in os.path.splitext(os.path.basename(paths[0]))[0].split("-")[-4:]]
        with open(path, "r") as f:
            log_text = f.read()
            if re.match("Average results:", log_text) is None:
                finished_hparams.append([a, g, l])
                avg_results_text = log_text.split('Average results: ')[1][:-1]
                avg_results = literal_eval(avg_results_text)
                all_avg_results.append(avg_results)

    return all_avg_results, finished_hparams