import torch
import time
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import KFold
import numpy as np
from dataset import mydataset
from sklearn.metrics import roc_curve, auc
from sklearn import metrics


def update_params(pre_w, now_w, lamda, lr):
    keys = []
    for key in now_w.keys():
        keys.append(key)

    for i in range(len(keys)):
        now_w[keys[i]] = now_w[keys[i]].to("cpu") - lamda * lr * (now_w[keys[i]].to("cpu") - pre_w[keys[i]].to("cpu"))
    return now_w


class focal_loss(nn.Module):
    """
    focal_loss loss function
    """

    def __init__(self, alpha=0.25, gamma=2.0, num_classes=2, size_average=True):
        super(focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        else:
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)

        self.gamma = gamma

    def forward(self, preds, labels):
        BCE_loss = F.binary_cross_entropy_with_logits(preds, labels, reduction='none')
        pt = torch.exp(-BCE_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


# class ConsistencyRegularizationLoss(nn.Module):
#     def __init__(self):
#         super(ConsistencyRegularizationLoss, self).__init__()
#
#     def forward(self, outputs):
#         # outputs: 模型在无标签数据上的输出，维度为 (batch_size, output_dim)
#
#         # 计算输出之间的平方欧氏距离
#         num_samples = outputs.size(0)
#         pairwise_distances = torch.cdist(outputs, outputs, p=2.0)  # 计算pairwise欧氏距离
#         consistency_loss = (pairwise_distances ** 2).sum() / (num_samples * (num_samples - 1))
#
#         return consistency_loss

class Client(object):
    def __init__(self, model, cancer_name, device, alpha, gamma, lamda, learning_rate, id=-1):
        self.local_model = model
        self.client_id = id
        self.cancer_name = cancer_name
        self.device = device
        self.data_path = 'data/%s' % self.cancer_name
        self.dataset = mydataset(self.data_path)
        self.data = self.dataset[0].to(self.device)
        self.lamda = lamda
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.gamma = gamma

    def local_train(self, model):
        roc = np.zeros(shape=(1, 5))
        apruc = np.zeros(shape=(1, 5))
        new_weight = model.state_dict()
        params = update_params(pre_w=new_weight, now_w=self.local_model.state_dict(), lamda=self.lamda,
                               lr=self.learning_rate)
        self.local_model.load_state_dict(params)
        self.local_model.to(self.device)
        kf = KFold(n_splits=5, shuffle=True)
        critertion = focal_loss(alpha=self.alpha, gamma=self.gamma)
        optimizer = optim.Adam(params=self.local_model.parameters(), lr=self.learning_rate)
        self.local_model.train()
        start_time = time.time()

        for e in range(3):
            train_loss = 0.0
            test_loss = 0.0
            labeled_indices = torch.where((self.data.y == 1) | (self.data.y == 0))[0]
            for fold, (train_index, test_index) in enumerate(kf.split(labeled_indices)):
                self.local_model.zero_grad()
                optimizer.zero_grad()

                predict_value = self.local_model(self.data, self.device).to(torch.device('cpu'))

                train_indices = labeled_indices[train_index]
                test_indices = labeled_indices[test_index]
                train_predict = torch.sigmoid(predict_value[train_indices])
                loss = critertion(train_predict, self.data.y[train_indices].to('cpu', dtype=torch.float32))
                train_loss += loss.item()
                loss.backward()
                optimizer.step()

                # Test Model
                self.local_model.eval()
                model.eval()
                with torch.no_grad():
                    predict = torch.sigmoid(predict_value[test_indices])
                    pred = predict.cpu().detach().numpy()
                    loss = critertion(predict, self.data.y[test_indices].to('cpu', dtype=torch.float32))
                    test_loss += loss.item()
                    fpr, tpr, _ = roc_curve(self.data.y.squeeze(0)[test_indices].detach().cpu().numpy(), pred)
                    roc[0][fold] = auc(fpr, tpr)
                    precision, recall, _ = metrics.precision_recall_curve(
                        self.data.y.squeeze(0)[test_indices].detach().cpu().numpy(), pred)
                    apruc[0][fold] = metrics.average_precision_score(
                        self.data.y.squeeze(0)[test_indices].detach().cpu().numpy(), pred)

        end_time = time.time()
        local_weight = self.local_model.state_dict()
        print(
            "Client {}:Train Loss is : {:02.4f}; Test Loss is : {:02.4f},roc is : {:02.4f},ap is :{:02.4f} ,Spend time: {:02.2f} mins".format(
                self.cancer_name, train_loss / 5, test_loss / 5, roc.mean(),apruc.mean (),(end_time - start_time) / 60))

        return train_loss / 5, test_loss / 5, local_weight

    def results(self, model):
        self.local_model.to(self.device)
        model.to(self.device)
        self.data = self.data.to(self.device)

        self.local_model.zero_grad()
        local_results = self.local_model(self.data, self.device).to(torch.device('cpu'))
        local_results = torch.sigmoid(local_results)

        model.zero_grad()
        server_results = model(self.data, self.device).to(torch.device('cpu'))
        server_results = torch.sigmoid(server_results)

        local_results = local_results.cpu().detach().numpy()
        server_results = server_results.cpu().detach().numpy()

        np.savetxt('results/gene_score/%s_local_result.csv' % self.cancer_name, local_results, delimiter=',', fmt='%f')
        np.savetxt('results/gene_score/%s_server_result.csv' % self.cancer_name, server_results, delimiter=',',
                   fmt='%f')
