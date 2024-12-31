import copy
from server import *
from client import *
import models


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            # print('done')
            w_avg[k] += w[i][k]
        # 通过将w_avg[k]除以参与者数量len(w)来计算平均值。最终得到的w_avg即为全局模型的参数
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def pFedMe(pre_w, w, beta):
    keys = []
    for key in w.keys():
        keys.append(key)

    for i in range(len(keys)):
        w[keys[i]] = (1 - beta) * pre_w[keys[i]].to("cuda") + beta * w[keys[i]]
    return w


def run_FLDL_SCdriver():
    parameter = []
    dict = {}
    dict['aphla'] = float(0.9658)
    dict['beta'] = float(0.0452)
    dict['dropout'] = float(0.3)
    dict['gamma'] = float(0.8253)
    dict['local_lamda'] = int(28)
    dict['learning_rate'] = float(0.0057)
    parameter.append(dict)
    for p in parameter:
        cancer_types = ['BLCA', 'BRCA', 'COAD', 'ESCA', 'GBM', 'HNSC',
                        'KIRC', 'KIRP', 'LIHC', 'LUAD', 'LUSC', 'OV',
                        'PAAD', 'PRAD', 'SARC', 'STAD']

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        server = Server(dropout_rate=p['dropout'], gamma=p['gamma'], alpha=p['aphla'],
                        learning_rate=p['learning_rate'])
        clients = []  # Definition of the client list
        id = 0

        for cancer in cancer_types:
            clients.append(
                Client(model=models.ChebNet(in_c=64, hid_c_1=300, hid_c_2=100, out_c=1, K=2,
                                            dropout_rate=p['dropout']), cancer_name=cancer,

                       device=device, alpha=p['aphla'], gamma=p['gamma'], lamda=p['local_lamda'],
                       learning_rate=p['learning_rate'], id=id))
            id += 1

        # Global Training
        Epoch = 500
        w_locals = []
        start_time = time.time()
        for e in range(Epoch):
            print('---------------------------Epoch {} Begining-------------------------'.format(e + 1))
            client_names = []
            # Traverse the selected clients and perform local training on each client
            for c in clients:
                client_names.append(c.cancer_name)
                train_loss, test_loss, w = c.local_train(server.global_model)
                w_locals.append(copy.deepcopy(w))
                # Update the global weights based on the parameter difference dictionary returned by the clients
            pre_w = copy.deepcopy(server.global_model.state_dict())
            now_w = FedAvg(w_locals)
            w_glob = pFedMe(pre_w=pre_w, w=now_w, beta=p['beta'])
            server.global_model.load_state_dict(w_glob)
            _ = server.pancancer_pro(e)
            print(
                '---------------------------Epoch {} Ending Spend time : {:02.4f} mins-------------------------'.format(
                    e + 1, (time.time() - start_time) / 60))
        # for c in clients:
        #     c.results(server.global_model)


if __name__ == '__main__':
    run_FLDL_SCdriver()
