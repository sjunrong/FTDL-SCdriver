import torch
import numpy as np
from torch_geometric.data import Dataset, Data
import os
from sklearn.preprocessing import StandardScaler
# 定义自己的数据集类
class mydataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(mydataset, self).__init__(root, transform, pre_transform)

    # Original file location
    @property
    def raw_file_names(self):
        return ['labels.csv', 'gene_feature.npz','new_graph.csv','import_feature.npz']

    # 文件保存位置
    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass

    # Data processing logic
    def process(self):
        labels=np.loadtxt(self.raw_paths[0],delimiter=',',dtype=str)[1:,]
        labels=labels[:,1:].astype(int)
        y=torch.tensor(labels)
        x=np.load(self.raw_paths[1])['data']
        important_feature=np.load(self.raw_paths[3])['data']
        important_feature=torch.tensor(important_feature,dtype=torch.float32)
        scaler = StandardScaler()
        x=scaler.fit_transform(x)
        x = torch.tensor(x, dtype=torch.float32)
        x=torch.cat((x,important_feature),1)

        edges = np.loadtxt(self.raw_paths[2], delimiter=',',dtype=np.int32)
        edge_str = edges[:,:1]
        edge_end = edges[:,1:]
        edge_index = torch.tensor([edge_str, edge_end], dtype=torch.long)
        edge_index = edge_index.squeeze()
        data = Data(x=x, edge_index=edge_index, y=y)

        torch.save(data, os.path.join(self.processed_dir, f'data.pt'))

    def len(self):
        return 1

    # Define a method to retrieve data
    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data.pt'))
        return data
