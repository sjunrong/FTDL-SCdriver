from sklearn.metrics import roc_curve, auc
from optuna_server import *
from optuna_client import *
import optuna_models
import optuna
import os
import time

class Utils:
    @staticmethod
    def create_folder(path):
        try:
            if not path:
                raise ValueError("Path cannot be empty")
            if os.path.exists(path):
                print("Folder already exists")
                return False, "Folder already exists"
            os.makedirs(path)
            return True, "Folder created successfully"
        except Exception as e:
            error_message = f"Failed to create folder: {e}"
            print(error_message)
            return False, error_message

def objective(trial):
    params = {
        'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
        'optimizer': trial.suggest_categorical('optimizer', ['Adam']),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5, step=0.1),
        'local_lamda': trial.suggest_int('local_lamda', 10, 50),
        'alpha': trial.suggest_float('alpha', 0.1, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'beta': trial.suggest_float('beta', 0, 1)
    }

    cancer_types = ['BLCA', 'BRCA', 'COAD', 'ESCA', 'GBM', 'HNSC',
                    'KIRC', 'KIRP', 'LIHC', 'LUAD', 'LUSC', 'OV',
                    'PAAD', 'PRAD', 'SARC', 'STAD'
                    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    server = Server(dropout_rate=params['dropout'])
    clients = []
    id = 0
    for cancer in cancer_types:
        clients.append(
            Client(model=optuna_models.ChebNet(in_c=64, hid_c_1=300, hid_c_2=100, out_c=1,dropout=params['dropout'] ,K=2), cancer_name=cancer,lr=params['lr'],
                   optimizer=params['optimizer'],local_lamda=params['local_lamda'],
                   alpha=params['alpha'],gamma=params['gamma'],
                   device=device, id=id))
        id += 1
        # 全局训练
    Epoch = 500
    w_locals = []
    for e in range(Epoch):
        start_time = time.time()
        print('---------------------------Epoch {} Begining-------------------------'.format(e + 1))
        # print(' ')
        # print(' ')
        results = []
        client_names = []
        # 遍历选中的客户端，每个客户端本地进行训练
        for c in clients:
            client_names.append(c.cancer_name)

            targets, Score, train_loss, test_loss, w = c.local_train(server.global_model)
            result = [targets, Score, train_loss, test_loss]
            results.append(result)
            w_locals.append(copy.deepcopy(w))
        t_loss = []
        loss = []


        for name, cancer_result in zip(client_names, results):
            targets, Score, train_loss, test_loss = cancer_result
            if name == 'pancancer':
                fpr, tpr, _ = roc_curve(targets, Score)
                roc_auc = auc(fpr, tpr)
                trial.report(roc_auc, e)
                pancaner_auc=roc_auc
            t_loss.append(train_loss)
            loss.append(test_loss)



        pre_w = copy.deepcopy(server.global_model.state_dict())
        now_w = server.FedAvg(w_locals)
        w_glob = server.pFedMe(pre_w=pre_w, w=now_w, beta=params['beta'])
        server.global_model.load_state_dict(w_glob)


    return pancaner_auc

if __name__ == '__main__':
    storage_name = 'sqlite:///optuna.db'
    study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=200),
        direction='maximize', study_name='Pancancer',storage=storage_name
    )

    study.optimize(objective, n_trials=100)
    best_trial = study.best_trial
    best_value = best_trial.value

    print('\n\n best_value = '+str(best_value))
    print('best_params:')
    print(best_trial)


