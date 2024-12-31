import models
from client import *
from dataset import mydataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Server(object):
    def __init__(self, dropout_rate, gamma, alpha,
                 learning_rate):
        self.global_model = models.ChebNet(in_c=64, hid_c_1=300, hid_c_2=100, out_c=1, K=2,
                                           dropout_rate=dropout_rate).to(device)
        self.data_path = 'data/server_data'
        self.dataset = mydataset(self.data_path)
        self.data = self.dataset[0].to(device)
        self.test_data = mydataset('data/pancancer')[0].to(device)
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.gamma = gamma

    def model_aggregate(self, weight_accumulator):
        # Traverse the global model on the server
        for name, data in self.global_model.state_dict().items():

            update_per_layer = weight_accumulator[name] * 0.5
            # Move the update_per_layer tensor to the same device as the data tensor
            update_per_layer = update_per_layer.to(data.device)
            # add
            if data.type() != update_per_layer.type():
                # Since the type of update_per_layer is floatTensor, it is converted to the model's LongTensor (loss of precision)

                data.add_(update_per_layer.to(torch.int64))
            else:
                data.add_(update_per_layer)

    # # The global model does not participate in the training
    # def pancancer_pro(self):
    #     Score = np.empty(shape=(1, 1))
    #     targets = np.empty(shape=(1, 1))
    #     kf = KFold(n_splits=5, shuffle=True)
    #     all_indices = torch.arange(len(self.data.y))
    #     labeled_indices = torch.where((self.data.y == 1) | (self.data.y == 0))[0]
    #     unlabeled_indices = torch.tensor(list(set(all_indices.tolist()) - set(labeled_indices.tolist())))
    #     for fold, (train_index, test_index) in enumerate(kf.split(labeled_indices)):
    #         test_indices = labeled_indices[test_index]
    #
    #
    #         # Test Model
    #         self.global_model.eval()
    #         with torch.no_grad():
    #             predict_value=self.global_model(self.data,device)
    #             predict = torch.sigmoid(predict_value[test_indices])
    #             pred = predict.cpu().detach().numpy()
    #             Score = np.concatenate((pred, Score))
    #             targets = np.concatenate(
    #                 (self.data.y.squeeze(0)[test_indices].detach().cpu().numpy(), targets))
    #             targets = targets[:targets.shape[0] - 1]
    #             Score = Score[:Score.shape[0] - 1]
    #     return Score, targets
    # The global model participates in the training
    def pancancer_pro(self, epoch):
        roc = np.zeros(shape=(1, 5))
        apruc = np.zeros(shape=(1, 5))
        train_loss = 0.0
        test_loss = 0.0
        kf = KFold(n_splits=5, shuffle=True)
        critertion = focal_loss(alpha=self.alpha, gamma=self.gamma)
        optimizer = optim.Adam(params=self.global_model.parameters(), lr=self.learning_rate)
        labeled_indices = torch.where((self.data.y == 1) | (self.data.y == 0))[0]
        for fold, (train_index, test_index) in enumerate(kf.split(labeled_indices)):
            self.global_model.train()
            self.global_model.zero_grad()
            optimizer.zero_grad()
            predict_value = self.global_model(self.data, device).to(torch.device('cpu'))
            train_indices = labeled_indices[train_index]
            test_indices = labeled_indices[test_index]
            train_predict = torch.sigmoid(predict_value[train_indices])

            loss = critertion(train_predict, self.data.y[train_indices].to('cpu', dtype=torch.float32))
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            # Test Model
            self.global_model.eval()
            with torch.no_grad():
                predict = torch.sigmoid(predict_value[test_indices])
                pred = predict.cpu().detach().numpy()
                fpr, tpr, _ = roc_curve(self.data.y.squeeze(0)[test_indices].detach().cpu().numpy(), pred)
                roc[0][fold] = auc(fpr, tpr)
                precision, recall, _ = metrics.precision_recall_curve(
                    self.data.y.squeeze(0)[test_indices].detach().cpu().numpy(), pred)
                apruc[0][fold] = metrics.average_precision_score(
                    self.data.y.squeeze(0)[test_indices].detach().cpu().numpy(), pred)
                loss = critertion(predict, self.data.y[test_indices].to('cpu', dtype=torch.float32))
                test_loss += loss.item()
        print("Server:Train Loss is : {:02.4f}, Test Loss is : {:02.4f} ,roc is : {:02.4f}, ap is : {:02.4f}".format(
             train_loss / 5, test_loss / 5,roc.mean(),apruc.mean()))
        return train_loss / 5
