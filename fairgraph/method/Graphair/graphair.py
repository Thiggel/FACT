import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
from fairgraph.utils.utils import scipysp_to_pytorchsp, accuracy, fair_metric

class Graphair(nn.Module):
    r'''
        This class implements the Graphair model

        :param aug_model: The augmentation model g described in the `paper <https://openreview.net/forum?id=1_OGWcP1s9w>`_ used for automated graph augmentations
        :type aug_model: :obj:`torch.nn.Module`

        :param f_encoder: The represnetation encoder f described in the `paper <https://openreview.net/forum?id=1_OGWcP1s9w>`_ used for contrastive learning
        :type f_encoder: :obj:`torch.nn.Module`

        :param sens_model: The adversary model k described in the `paper <https://openreview.net/forum?id=1_OGWcP1s9w>`_ used for adversarial learning
        :type sens_model: :obj:`torch.nn.Module`

        :param classifier_model: The classifier used to predict the sensitive label of nodes on the augmented graph data.
        :type classifier_model: :obj:`torch.nn.Module`

        :param lr: Learning rate for aug_model, f_encoder and sens_model. Defaults to 1e-4
        :type lr: float,optional

        :param weight_decay: Weight decay for regularization. Defaults to 1e-5
        :type weight_decay: float,optional

        :param alpha: The hyperparameter alpha used in the `paper <https://openreview.net/forum?id=1_OGWcP1s9w>`_ to scale adversarial loss component. Defaults to 20.0
        :type alpha: float,optional

        :param beta: The hyperparameter beta used in the `paper <https://openreview.net/forum?id=1_OGWcP1s9w>`_ to scale contrastive loss component. Defaults to 0.9
        :type beta: float,optional

        :param gamma: The hyperparameter gamma used in the `paper <https://openreview.net/forum?id=1_OGWcP1s9w>`_ to scale reconstruction loss component. Defaults to 0.7
        :type gamma: float,optional

        :param lam: The hyperparameter lambda used in the `paper <https://openreview.net/forum?id=1_OGWcP1s9w>`_ to compute reconstruction loss component. Defaults to 1.0
        :type lam: float,optional

        :param dataset: The name of the dataset being used. Used only for the model's output path. Defaults to 'POKEC'
        :type dataset: str,optional

        :param num_hidden: The input dimension for the MLP networks used in the model. Defaults to 64
        :type num_hidden: int,optional

        :param num_proj_hidden: The output dimension for the MLP networks used in the model. Defaults to 64
        :type num_proj_hidden: int,optional

    '''
    def __init__(self, aug_model, f_encoder, sens_model, classifier_model, k_lr=1e-4,
                 c_lr=1e-3, g_lr=1e-4, g_warmup_lr=1e-3, f_lr=1e-4,
                 weight_decay=1e-5, alpha=10, beta=0.1, gamma=0.5, lam=0.5, temperature=0.07,
                 num_hidden=64, num_proj_hidden=64, dataset='POKEC', device='cpu',
                 batch_size=None, checkpoint_path='./checkpoint/'):
        super(Graphair, self).__init__()
        self.device = device
        self.checkpoint_path = checkpoint_path

        self.aug_model = aug_model
        self.f_encoder = f_encoder
        self.sens_model = sens_model
        self.classifier = classifier_model
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.dataset = dataset
        self.lam = lam
        self.temperature = temperature
        self.batch_size = batch_size

        self.criterion_sens = nn.BCEWithLogitsLoss()
        self.criterion_cont= nn.CrossEntropyLoss()
        self.criterion_recons = nn.MSELoss()

        self.optimizer_s = torch.optim.Adam(self.sens_model.parameters(), lr=k_lr, weight_decay=weight_decay)

        FG_params = [{'params': self.aug_model.parameters(), 'lr': g_lr} ,  {'params': self.f_encoder.parameters(), 'lr': f_lr}]
        self.optimizer = torch.optim.Adam(FG_params, weight_decay=weight_decay)

        self.optimizer_aug = torch.optim.Adam(self.aug_model.parameters(), lr=g_warmup_lr, weight_decay=weight_decay)
        self.optimizer_enc = torch.optim.Adam(self.f_encoder.parameters(), lr=f_lr, weight_decay=weight_decay)


        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

        self.optimizer_classifier = torch.optim.Adam(self.classifier.parameters(),
                            lr=c_lr, weight_decay=weight_decay)
    
    def projection(self, z):
        z = F.elu(self.fc1(z))
        return self.fc2(z)
    
    def info_nce_loss_2views(self, features):
        
        batch_size = int(features.shape[0] / 2)

        labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)
        
        logits = logits / self.temperature
        return logits, labels


    def forward(self, adj, x):
        assert sp.issparse(adj)
        if not isinstance(adj, sp.coo_matrix):
            adj = sp.coo_matrix(adj)
        adj.setdiag(1)
        degrees = np.array(adj.sum(1))
        degree_mat_inv_sqrt = sp.diags(np.power(degrees, -0.5).flatten())
        adj_norm = degree_mat_inv_sqrt @ adj @ degree_mat_inv_sqrt
        adj_norm = scipysp_to_pytorchsp(adj_norm)
    
        adj = adj_norm.to(self.device)
        return self.f_encoder(adj,x)
    
    def fit_batch_GraphSAINT(self, epochs, adj, x, sens, idx_sens, minibatch, warmup=None, adv_epoches=10, verbose=False):
        assert sp.issparse(adj)
        if not isinstance(adj, sp.coo_matrix):
            adj = sp.coo_matrix(adj)
        adj.setdiag(1)
        adj_orig = sp.csr_matrix(adj)
        norm_w = adj_orig.shape[0]**2 / float((adj_orig.shape[0]**2 - adj_orig.sum()) * 2)

        idx_sens = idx_sens.cpu().numpy()

        if warmup:
            for _ in range(warmup):

                node_subgraph, adj, _ = minibatch.one_batch(mode='train')
                adj = adj.to(self.device)
                edge_label = torch.FloatTensor(adj_orig[node_subgraph][:,node_subgraph].toarray()).to(self.device)

                adj_aug, x_aug, adj_logits = self.aug_model(adj, x[node_subgraph], adj_orig = edge_label)
                edge_loss = norm_w * F.binary_cross_entropy_with_logits(adj_logits, edge_label)


                feat_loss =  self.criterion_recons(x_aug, x[node_subgraph])
                recons_loss =  edge_loss + self.beta * feat_loss

                self.optimizer_aug.zero_grad()
                with torch.autograd.set_detect_anomaly(True):
                    recons_loss.backward(retain_graph=True)
                self.optimizer_aug.step()

                if verbose:
                    print(
                    'edge reconstruction loss: {:.4f}'.format(edge_loss.item()),
                    'feature reconstruction loss: {:.4f}'.format(feat_loss.item()),
                    )

        for epoch_counter in range(epochs):
            ### generate fair view
            node_subgraph, adj, norm_loss_subgraph = minibatch.one_batch(mode='train')
            adj = adj.to(self.device)
            norm_loss_subgraph = norm_loss_subgraph.to(self.device)

            edge_label = torch.FloatTensor(adj_orig[node_subgraph][:,node_subgraph].toarray()).to(self.device)
            adj_aug, x_aug, adj_logits = self.aug_model(adj, x[node_subgraph], adj_orig = edge_label)
            # print("aug done")

            ### extract node representations
            h = self.projection(self.f_encoder(adj, x[node_subgraph]))
            h_prime = self.projection(self.f_encoder(adj_aug, x_aug))
            # print("encoder done")

            ### update sens model
            adj_aug_nograd = adj_aug.detach()
            x_aug_nograd = x_aug.detach()

            mask = np.in1d(node_subgraph, idx_sens)

            if (epoch_counter == 0):
                sens_epoches = adv_epoches * 10
            else:
                sens_epoches = adv_epoches
            for _ in range(sens_epoches):

                s_pred , _  = self.sens_model(adj_aug_nograd, x_aug_nograd)
                senloss = torch.nn.BCEWithLogitsLoss(weight=norm_loss_subgraph,reduction='sum')(s_pred[mask].squeeze(),sens[node_subgraph][mask].float())

                self.optimizer_s.zero_grad()
                senloss.backward()
                self.optimizer_s.step()

            s_pred , _  = self.sens_model(adj_aug, x_aug)
            senloss = torch.nn.BCEWithLogitsLoss(weight=norm_loss_subgraph,reduction='sum')(s_pred[mask].squeeze(),sens[node_subgraph][mask].float())

            ## update aug model
            logits, labels = self.info_nce_loss_2views(torch.cat((h, h_prime), dim = 0))
            contrastive_loss = (torch.nn.CrossEntropyLoss(reduction='none')(logits, labels) * norm_loss_subgraph.repeat(2)).sum() 

            ## update encoder
            edge_loss = norm_w * F.binary_cross_entropy_with_logits(adj_logits, edge_label)


            feat_loss =  self.criterion_recons(x_aug, x[node_subgraph])
            recons_loss =  edge_loss + self.lam * feat_loss

            loss = self.beta * contrastive_loss + self.gamma * recons_loss - self.alpha * senloss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if ((epoch_counter + 1) % 1000 == 0 and verbose):
                print('Epoch: {:04d}'.format(epoch_counter+1),
                'sens loss: {:.4f}'.format(senloss.item()),
                'contrastive loss: {:.4f}'.format(contrastive_loss.item()),
                'edge reconstruction loss: {:.4f}'.format(edge_loss.item()),
                'feature reconstruction loss: {:.4f}'.format(feat_loss.item()),
                )

        self._save_checkpoint()

    def fit_whole(self, epochs, adj, x,sens,idx_sens,warmup=None, adv_epoches=1, verbose=False):
        assert sp.issparse(adj)
        if not isinstance(adj, sp.coo_matrix):
            adj = sp.coo_matrix(adj)
        adj.setdiag(1)
        adj_orig = scipysp_to_pytorchsp(adj).to_dense()
        norm_w = adj_orig.shape[0]**2 / float((adj_orig.shape[0]**2 - adj_orig.sum()) * 2)
        degrees = np.array(adj.sum(1))
        degree_mat_inv_sqrt = sp.diags(np.power(degrees, -0.5).flatten())
        adj_norm = degree_mat_inv_sqrt @ adj @ degree_mat_inv_sqrt
        adj_norm = scipysp_to_pytorchsp(adj_norm)
        
        adj = adj_norm.to(self.device)
        
        if warmup:
            for _ in range(warmup):
                adj_aug, x_aug, adj_logits = self.aug_model(adj, x, adj_orig = adj_orig.to(self.device))
                edge_loss = norm_w * F.binary_cross_entropy_with_logits(adj_logits, adj_orig.to(self.device))

                feat_loss =  self.criterion_recons(x_aug, x)
                recons_loss =  edge_loss + self.beta * feat_loss

                self.optimizer_aug.zero_grad()
                with torch.autograd.set_detect_anomaly(True):
                    recons_loss.backward(retain_graph=True)
                self.optimizer_aug.step()

                if verbose:
                    print(
                    'edge reconstruction loss: {:.4f}'.format(edge_loss.item()),
                    'feature reconstruction loss: {:.4f}'.format(feat_loss.item()),
                    )

        for epoch_counter in range(epochs):
            # print(f"Epoch {epoch_counter}")
            ### generate fair view
            adj_aug, x_aug, adj_logits = self.aug_model(adj, x, adj_orig = adj_orig.to(self.device))

            adj_aug = adj_aug.to_sparse_coo()
            ### extract node representations
            h = self.projection(self.f_encoder(adj, x))
            h_prime = self.projection(self.f_encoder(adj_aug, x_aug))
            # print("encoder done")

            ## update sens model
            adj_aug_nograd = adj_aug.detach()
            x_aug_nograd = x_aug.detach()
            if (epoch_counter == 0):
                sens_epoches = adv_epoches * 10
            else:
                sens_epoches = adv_epoches
            for _ in range(sens_epoches):
                s_pred , _  = self.sens_model(adj_aug_nograd, x_aug_nograd)
                senloss = self.criterion_sens(s_pred[idx_sens],sens[idx_sens].unsqueeze(1).float())
                self.optimizer_s.zero_grad()
                senloss.backward()
                self.optimizer_s.step()
            s_pred , _  = self.sens_model(adj_aug, x_aug)
            senloss = self.criterion_sens(s_pred[idx_sens],sens[idx_sens].unsqueeze(1).float())

            ## update aug model
            logits, labels = self.info_nce_loss_2views(torch.cat((h, h_prime), dim = 0))
            contrastive_loss = self.criterion_cont(logits, labels)

            ## update encoder
            edge_loss = norm_w * F.binary_cross_entropy_with_logits(adj_logits, adj_orig.to(self.device))

            feat_loss =  self.criterion_recons(x_aug, x)
            recons_loss =  edge_loss + self.lam * feat_loss
            loss = self.beta * contrastive_loss + self.gamma * recons_loss - self.alpha * senloss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if verbose:
                print('Epoch: {:04d}'.format(epoch_counter+1),
                'sens loss: {:.4f}'.format(senloss.item()),
                'contrastive loss: {:.4f}'.format(contrastive_loss.item()),
                'edge reconstruction loss: {:.4f}'.format(edge_loss.item()),
                'feature reconstruction loss: {:.4f}'.format(feat_loss.item()),
                )
        self._save_checkpoint()
    

    def test(self, adj, features, labels, epochs, idx_train, idx_val, idx_test, sens, verbose=False):
        h = self.forward(adj, features)
        h = h.detach()

        acc_list = []
        dp_list = []
        eo_list = []

        for i in range(5):
            torch.manual_seed(i*10)
            np.random.seed(i*10)

            self.classifier.reset_parameters()
            # train classifier
            best_acc = 0.0
            best_test = 0.0
            for epoch in range(epochs):

                self.classifier.train()
                self.optimizer_classifier.zero_grad()
                output = self.classifier(h)
                loss_train = F.binary_cross_entropy_with_logits(output[idx_train], labels[idx_train].unsqueeze(1).float())
                acc_train = accuracy(output[idx_train], labels[idx_train])
                loss_train.backward()
                self.optimizer_classifier.step()
                            
                self.classifier.eval()
                output = self.classifier(h)
                acc_val = accuracy(output[idx_val], labels[idx_val])
                acc_test = accuracy(output[idx_test], labels[idx_test])

                parity_val, equality_val = fair_metric(output, idx_val, labels, sens)
                parity_test, equality_test = fair_metric(output, idx_test, labels, sens)
                if epoch%10==0 and verbose:
                    print("Epoch [{}] Test set results:".format(epoch),
                        "acc_test= {:.4f}".format(acc_test.item()),
                        "acc_val: {:.4f}".format(acc_val.item()),
                        "dp_val: {:.4f}".format(parity_val),
                        "dp_test: {:.4f}".format(parity_test),
                        "eo_val: {:.4f}".format(equality_val),
                        "eo_test: {:.4f}".format(equality_test), )
                if acc_val > best_acc:
                    best_acc = acc_val
                    best_test = acc_test
                    best_dp = parity_val
                    best_dp_test = parity_test
                    best_eo = equality_val
                    best_eo_test = equality_test

            if verbose:
                print("Optimization Finished!")
                print("Test results:",
                            "acc_test= {:.4f}".format(best_test.item()),
                            "acc_val: {:.4f}".format(best_acc.item()),
                            "dp_val: {:.4f}".format(best_dp),
                            "dp_test: {:.4f}".format(best_dp_test),
                            "eo_val: {:.4f}".format(best_eo),
                            "eo_test: {:.4f}".format(best_eo_test),)
        
            acc_list.append(best_test.item())
            dp_list.append(best_dp_test)
            eo_list.append(best_eo_test)
        
        average_results = {
            "acc": {"mean": np.mean(acc_list), "std": np.std(acc_list)},
            "dp": {"mean": np.mean(dp_list), "std": np.std(dp_list)},
            "eo": {"mean": np.mean(eo_list), "std": np.std(eo_list)}
        }
        print("Avg results:",
                    "acc: {:.4f} std: {:.4f}".format(average_results["acc"]["mean"], average_results["acc"]["std"]), 
                    "dp: {:.4f} std: {:.4f}".format(average_results["dp"]["mean"], average_results["dp"]["std"]),
                    "eo: {:.4f} std: {:.4f}".format(average_results["eo"]["mean"], average_results["eo"]["std"]),)
        return average_results

    def _save_checkpoint(self):
        os.makedirs(self.checkpoint_path, exist_ok=True)
        save_path = f"{self.checkpoint_path}graphair_{self.dataset}_alpha{self.alpha}_beta{self.beta}_gamma{self.gamma}_lambda{self.lam}"
        if self.batch_size:
            save_path += "_batch_size{}"
        torch.save(self.state_dict(), save_path)
