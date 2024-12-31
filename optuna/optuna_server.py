import torch
import optuna_models
import copy
class Server(object):
    def __init__(self,dropout_rate):
        self.global_model=optuna_models.ChebNet(in_c=64,hid_c_1=300,hid_c_2=100,out_c=1,dropout=dropout_rate,K=2)

    def FedAvg(self,w):
        w_avg = copy.deepcopy(w[0])
        for k in w_avg.keys():
            for i in range(1, len(w)):
                # print('done')
                w_avg[k] += w[i][k]
            # 通过将w_avg[k]除以参与者数量len(w)来计算平均值。最终得到的w_avg即为全局模型的参数
            w_avg[k] = torch.div(w_avg[k], len(w))
        return w_avg


    def pFedMe(self,pre_w, w, beta):
        keys = []
        for key in w.keys():
            keys.append(key)

        for i in range(len(keys)):
            w[keys[i]] = (1 - beta) * pre_w[keys[i]].to("cuda") + beta * w[keys[i]]
        return w
