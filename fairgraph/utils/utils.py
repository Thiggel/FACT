import numpy as np
import scipy.sparse as sp
import torch
import matplotlib.pyplot as plt

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

def get_pareto_front(self, data, fairness_metric='dp'):
    accs = np.array([d['accuracy']['mean'] for d in data])              # higher is better
    fair_metrics = np.array([d[fairness_metric]['mean'] for d in data]) # lower is better

    pareto_front = []
    for i, (acc, fair_metric) in enumerate(zip(accs, fair_metrics)):
        better_idxs = np.where((accs >= acc) & (fair_metrics <= fair_metric))[0]
        
        # remove i from idxs
        better_idxs = np.delete(better_idxs, np.where(better_idxs == i))

        if len(better_idxs) == 0 or np.all((accs[better_idxs] == acc) & (fair_metrics[better_idxs] == fair_metric)): 
            pareto_front.append(data[i])

    return pareto_front

def visualize_pareto_front(
    self,
    data,
    fairness_metric='dp',
    filename='pareto_front.png'
):
    sorted_data = sorted(
        data,
        key=lambda x: (-x['accuracy']['mean'], x[fairness_metric]['mean'])
    )

    accuracy = [item['accuracy']['mean'] for item in sorted_data]
    dp = [item[fairness_metric]['mean'] for item in sorted_data]

    plt.figure(figsize=(10, 6))
    plt.scatter(dp, accuracy, color='b')
    plt.plot(dp, accuracy, color='r')

    plt.xlabel(fairness_metric.upper())
    plt.ylabel('Accuracy')
    plt.title('Pareto Front')

    plt.savefig(filename)