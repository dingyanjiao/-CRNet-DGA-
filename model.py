import torch
import torch.nn.functional as F
import crnet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CRNet(torch.nn.Module):
    def __init__(self, sequence_length=222, layers=4, lcffn_size=4096, embedding_dim=128, heads=4):
        super(CRNet, self).__init__()
        self.embedding = torch.nn.Embedding(40, embedding_dim)
        self.retnet = crnet.CRNet(layers, embedding_dim, lcffn_size, heads, double_v_dim=True).to(device)
        
        self.conv2 = torch.nn.Conv1d(in_channels=embedding_dim, out_channels=128, kernel_size=2, padding=1)
        self.conv3 = torch.nn.Conv1d(in_channels=embedding_dim, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv1d(in_channels=embedding_dim, out_channels=128, kernel_size=4, padding=2)
        
        self.global_pool = torch.nn.AdaptiveMaxPool1d(1)
        self.dropout = torch.nn.Dropout(0.2)
        
        self.fc = torch.nn.Linear(embedding_dim + 128 * 3, 2)
    
    def forward(self, url):
        embedding = self.embedding(url).transpose(1, 2)  # [batch_size, embedding_dim, sequence_length]
        
        cnn_out2 = self.global_pool(F.relu(self.conv2(embedding))).squeeze(-1)
        cnn_out3 = self.global_pool(F.relu(self.conv3(embedding))).squeeze(-1)
        cnn_out4 = self.global_pool(F.relu(self.conv4(embedding))).squeeze(-1)
        cnn_out = torch.cat([cnn_out2, cnn_out3, cnn_out4], dim=1)
        
        out = self.retnet(embedding.transpose(1, 2))  # [batch_size, sequence_length, embedding_dim]
        out = out[:, -1, :]
        
        out = torch.cat([out, cnn_out], dim=1)
        out = self.dropout(out)
        
        out = self.fc(out)
        scores = torch.nn.Sigmoid(out, dim=1)
        return scores
