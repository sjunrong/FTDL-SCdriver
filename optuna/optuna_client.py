import torch

import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import KFold
import numpy as np


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


def update_params(pre_w, now_w, lamda, lr):
    keys = []
    for key in now_w.keys():
        keys.append(key)

    for i in range(len(keys)):
        # now_w[keys[i]] = now_w[keys[i]].to("cpu") - lamda * lr * ( now_w[keys[i]].to("cpu")-pre_w[i].to("cpu"))
        now_w[keys[i]] = now_w[keys[i]].to("cpu") - lamda * lr * (now_w[keys[i]].to("cpu") - pre_w[keys[i]].to("cpu"))
    return now_w


class Client(object):
    def __init__(self, model, cancer_name, lr, optimizer, local_lamda, alpha, gamma, device, id=-1):
        self.local_model = model
        self.client_id = id
        self.cancer_name = cancer_name
        self.device = device
        self.data_path = 'data/%s' % self.cancer_name
        self.dataset = mydataset(self.data_path)
        self.data = self.dataset[0].to(self.device)
        self.gamma = gamma
        self.optimizer = optimizer
        self.alpha = alpha
        self.lamda = local_lamda
        self.learning_rate = lr

    def local_train(self, model):

        new_weight = model.state_dict()
        params = update_params(pre_w=new_weight, now_w=self.local_model.state_dict(), lamda=self.lamda,
                               lr=self.learning_rate)
        self.local_model.load_state_dict(params)
        self.local_model.to(self.device)
        kf = KFold(n_splits=5, shuffle=True)
        critertion = focal_loss(alpha=self.alpha, gamma=self.gamma)
        optimizer = getattr(optim, self.optimizer)(params=self.local_model.parameters(), lr=self.learning_rate)
        self.local_model.train()
        for e in range(3):
            Score = np.empty(shape=(1, 1))
            targets = np.empty(shape=(1, 1))
            train_loss = 0.0
            test_loss = 0.0
            all_indices = torch.arange(len(self.data.y))
            labeled_indices = torch.where((self.data.y == 1) | (self.data.y == 0))[0]
            for fold, (train_index, test_index) in enumerate(kf.split(labeled_indices)):
                self.local_model.zero_grad()
                optimizer.zero_grad()

                predict_value = self.local_model(self.data, self.device).to(torch.device('cpu'))

                train_indices = labeled_indices[train_index]
                test_indices = labeled_indices[test_index]
                train_predict = torch.sigmoid(predict_value[train_indices])
                loss_labeled = critertion(train_predict, self.data.y[train_indices].to('cpu', dtype=torch.float32))
                loss = loss_labeled
                train_loss += loss.item()
                loss.backward()
                optimizer.step()

                # Test Model
                self.local_model.eval()
                with torch.no_grad():
                    predict = torch.sigmoid(predict_value[test_indices])
                    pred = predict.cpu().detach().numpy()
                    Score = np.concatenate((pred, Score))
                    targets = np.concatenate(
                        (self.data.y.squeeze(0)[test_indices].detach().cpu().numpy(), targets))
                    loss = critertion(predict, self.data.y[test_indices].to('cpu', dtype=torch.float32))
                    test_loss += loss.item()
        targets = targets[:targets.shape[0] - 1]
        Score = Score[:Score.shape[0] - 1]

        local_weight = self.local_model.state_dict()

        return targets, Score, train_loss / 5, test_loss / 5, local_weight
