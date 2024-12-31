import torch.nn as nn
import torch.optim as optim
from models import MultiHeadSelfAttention
import numpy as np


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),  # Adjust dim_v to be the same as input_dim
            nn.ReLU()  # You can also use other activation functions; here, ReLU is used as an example
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


cancer_types = ['ACC', 'BLCA', 'BRCA', 'CHOL', 'COAD', 'DLBC', 'ESCA', 'GBM', 'HNSC',
                'KICH', 'KIRC', 'KIRP', 'LIHC', 'LUAD', 'LUSC', 'MESO', 'OV',
                'PAAD', 'PCPG', 'PRAD', 'SARC', 'STAD', 'TGCT', 'THCA',
                'UCS', 'UVM', 'pancancer']
for cancer in cancer_types:
    information = np.load(file='Trans_data_extract/train_and_test_data/%s/imformation.npz' % cancer)
    information = information['data']
    num_nodes = int(information[0][1])
    train_graph_path = './Trans_data_extract/train_and_test_data/%s/graph.csv' % cancer
    train_data_features_path = './Trans_data_extract/train_and_test_data/%s/gene_feature.npz' % cancer
    train_labels_path = './Trans_data_extract/train_and_test_data/%s/labels.csv' % cancer
    train_data = LoadData(data_path=[train_graph_path, train_data_features_path, train_labels_path],
                          num_nodes=num_nodes)
    train_loader = DataLoader(train_data, shuffle=False)
    input_dim = 46
    hidden_dim = 18
    out_dim = 46
    autoencoder = AutoEncoder(input_dim, hidden_dim, out_dim)
    attention_model = MultiHeadSelfAttention(dim_in=input_dim, dim_k=18, dim_v=18)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(autoencoder.parameters()) + list(attention_model.parameters()), lr=0.001)

    Epoch = 400

    for epoch in range(Epoch):
        total_loss = 0.0
        for data in train_loader:
            x = data['gene_x'].squeeze()
            optimizer.zero_grad()
            encoded = autoencoder.encoder(x)
            decoded = autoencoder.decoder(encoded)

            # Pass decoder output through attention model
            reconstructed = attention_model(decoded)

            loss = criterion(reconstructed, encoded)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

    np.savez(file='./Trans_data_extract/train_and_test_data/%s/import_feature' % cancer,
             data=reconstructed.cpu().detach().numpy())
    print(cancer)
