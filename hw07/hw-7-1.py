"""
input = torch.randn(10)
ret = torch.tensor([[-0.0282, -0.1428, 0.0725, 0.0197, 0.0999, 0.1904, -0.1933, -0.2090,
                     -0.0215, 0.0280, -0.0801, -0.1829, -0.0209, -0.0940, 0.1684, -0.1727,
                     0.0518, -0.0908, 0.1556, 0.2135, -0.0268, -0.0678, 0.2112, -0.0604,
                     -0.0122, -0.0312, -0.2091, 0.1665, -0.1171, 0.2131]])
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderAttentionDecoder:
    def __init__(self):
        self.bidirectional = True
        self.encoder = self.Encoder(10, 20, bidirectional=self.bidirectional)
        self.decoder = self.AttentionDecoder(20 * (1 + self.bidirectional), 25, 30)

    def forward(self, input):
        c = self.encoder
        a, b = c.forward(input, c.init_hidden())
        x = self.decoder
        y, z, w = x.forward(x.init_hidden(), torch.cat((a, a)), torch.zeros(1, 1, 30))
        return y

    class Encoder(nn.Module):
        def __init__(self, input_size, hidden_size, bidirectional=True):
            super().__init__()
            self.hidden_size = hidden_size
            self.input_size = input_size
            self.bidirectional = bidirectional

            self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=bidirectional)

        def forward(self, inputs, hidden):
            output, hidden = self.lstm(inputs.view(1, 1, self.input_size), hidden)
            return output, hidden

        def init_hidden(self):
            return (torch.zeros(1 + int(self.bidirectional), 1, self.hidden_size),
                    torch.zeros(1 + int(self.bidirectional), 1, self.hidden_size))

    class AttentionDecoder(nn.Module):

        def __init__(self, hidden_size, output_size, vocab_size):
            super().__init__()
            self.hidden_size = hidden_size
            self.output_size = output_size

            self.attn = nn.Linear(hidden_size + output_size, 1)
            self.lstm = nn.LSTM(hidden_size + vocab_size,
                                output_size)  # if we are using embedding hidden_size should be added with embedding of vocab size
            self.final = nn.Linear(output_size, vocab_size)

        def init_hidden(self):
            return (torch.zeros(1, 1, self.output_size),
                    torch.zeros(1, 1, self.output_size))

        def forward(self, decoder_hidden, encoder_outputs, input):
            weights = []
            for i in range(len(encoder_outputs)):
                weights.append(self.attn(torch.cat((decoder_hidden[0][0],
                                                    encoder_outputs[i]), dim=1)))
            normalized_weights = F.softmax(torch.cat(weights, 1), 1)

            attn_applied = torch.bmm(normalized_weights.unsqueeze(1),
                                     encoder_outputs.view(1, -1, self.hidden_size))

            input_lstm = torch.cat((attn_applied[0], input[0]),
                                   dim=1)  # if we are using embedding, use embedding of input here instead

            output, hidden = self.lstm(input_lstm.unsqueeze(0), decoder_hidden)

            output = self.final(output[0])

            return output, hidden, normalized_weights


model = EncoderAttentionDecoder()
print(model.forward(input))
