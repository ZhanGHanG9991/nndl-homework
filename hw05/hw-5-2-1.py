'''
x = torch.Tensor([[[0.6840, 0.1093],[1.0749, 2.3809],[0.3862, 0.4335]],
                            [[1.7795, 1.0825],[0.2621, 1.0084],[0.4832, 0.5558]]])
ret = torch.Tensor([[[0.0818, 0.1766],[0.0555, 0.1432],[0.0663, 0.1533]],
                                [[0.1013, 0.1704],[0.0852, 0.1404],[0.0932, 0.1542]], 
                                [[0.1215, 0.1684],[0.1122, 0.1458],[0.1178, 0.1591]]])
'''
import torch
import torch.nn as nn


class EncoderDecoder(nn.Module):
    def __init__(self, feature_dim=2, encoder_embedding_dim=8, decoder_embedding_dim=8, hidden_dim=16, n_layers=2,
                 target_len=3):
        super().__init__()
        self.target_len = target_len
        self.out_dim = feature_dim
        self.encoder = self.EncoderGRU(input_size=feature_dim, embedding_dim=encoder_embedding_dim,
                                       hidden_dim=hidden_dim, n_layers=n_layers)
        self.decoder = self.DecoderGRU(output_size=feature_dim, embedding_dim=decoder_embedding_dim,
                                       hidden_dim=hidden_dim, n_layers=n_layers)

    def forward(self, x):
        batch_size = x.shape[1]
        out = torch.zeros((self.target_len, batch_size, self.out_dim))
        _, hidden = self.encoder(x)
        de_in = x[-1, :]
        for i in range(self.target_len):
            de_out, hidden = self.decoder(de_in, hidden)
            out[i] = de_out
            de_in = de_out  # using predictions as the next input
        return out

    class EncoderGRU(nn.Module):
        def __init__(self, input_size=10, embedding_dim=8, hidden_dim=16, n_layers=2):
            super().__init__()
            self.embedding = nn.Sequential(nn.Linear(input_size, embedding_dim))
            self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=n_layers)

        def forward(self, x):
            embedded = self.embedding(x)
            out, hidden = self.gru(embedded)
            return out, hidden

    class DecoderGRU(nn.Module):
        def __init__(self, output_size, embedding_dim, hidden_dim, n_layers=1):
            super().__init__()
            self.embedding = nn.Sequential(nn.Linear(output_size, embedding_dim))
            self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=n_layers)
            self.fc = nn.Linear(hidden_dim, output_size)

        def forward(self, y, hidden):
            y = y.unsqueeze(0)
            embedded = self.embedding(y)
            y1, hidden = self.gru(embedded, hidden)
            out = self.fc(y1.squeeze(0))
            return out, hidden