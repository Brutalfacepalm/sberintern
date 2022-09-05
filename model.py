import torch
from torch import nn, autograd
import torch.nn.functional as F
import pickle
from utils import get_file, download_file_from_google_drive


INPUT_SIZE = 75
HIDDEN_SIZE = 512
NUM_LAYERS = 2
NUM_CLASSES = 1
HEADS = 12
TYPE_RNN = 'lstm'
BID = True
CONV_LAYERS = 2
KERNEL_CONV = 9


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BaselineModelLSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, bidirectional, type_rnn, heads, conv_layers,
                 kernel_conv):
        super(BaselineModelLSTM, self).__init__()

        self.device = DEVICE
        self.num_classes = num_classes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.type_rnn = type_rnn
        self.heads = heads
        self.conv_layers = conv_layers
        self.kernel_conv = kernel_conv

        self.body_rnn = nn.Sequential()

        if self.type_rnn == 'lstm':
            self.body_rnn.add_module('lstm', nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                                     num_layers=num_layers, batch_first=True,
                                                     bidirectional=self.bidirectional, dropout=0.2))
        elif self.type_rnn == 'gru':
            self.body_rnn.add_module('gru', nn.GRU(input_size=input_size, hidden_size=hidden_size,
                                                   num_layers=num_layers, batch_first=True,
                                                   bidirectional=self.bidirectional, dropout=0.2))
        if self.conv_layers:
            self.body_conv = nn.Sequential()

            for conv_l in range(1, self.conv_layers + 1):
                self.body_conv.add_module(f'conv_part_{conv_l + 1}',
                                          nn.Sequential(
                                              nn.Conv1d(self.input_size * conv_l, self.input_size * (conv_l + 1),
                                                        self.kernel_conv, 1, self.kernel_conv // 2),
                                              nn.BatchNorm1d(self.input_size * (conv_l + 1)),
                                              nn.ReLU(),
                                              ))
                self.conv_l_last = conv_l + 1

            self.body_conv.add_module(f'conv_part_last',
                                      nn.Sequential(
                                          nn.Conv1d(self.input_size * self.conv_l_last,
                                                    self.hidden_size, self.kernel_conv),
                                          nn.BatchNorm1d(self.hidden_size),
                                          nn.ReLU(),
                                          nn.AdaptiveMaxPool1d(self.num_layers)
                                          ))

            fc_input_size = hidden_size * (1 + (1 if self.bidirectional else 0)) * self.num_layers * 2 + hidden_size * (
                        1 + (1 if self.bidirectional else 0)) * 2
        else:
            fc_input_size = hidden_size * (1 + (1 if self.bidirectional else 0)) * self.num_layers * 2 + hidden_size * (
                        1 + (1 if self.bidirectional else 0))

        self.multi_head = nn.Sequential()

        for i in range(self.heads):
            self.multi_head.add_module(f'head_{i + 1}',
                                       nn.Sequential(nn.Linear(fc_input_size, fc_input_size // 8),
                                                     nn.SiLU(),
                                                     nn.Dropout(0.2),
                                                     nn.Linear(fc_input_size // 8, fc_input_size // 8 // 2),
                                                     nn.SiLU(),
                                                     nn.Dropout(0.2),
                                                     nn.Linear(fc_input_size // 8 // 2, num_classes)
                                                     ))

    def attention(self, lstm_output, h_state):
        hidden = h_state.reshape(-1, self.hidden_size * (1 + (1 if self.bidirectional else 0)), self.num_layers)
        attn_weights = torch.bmm(lstm_output, hidden)
        attn_weights = attn_weights.squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights)
        return context, soft_attn_weights

    def forward(self, x):

        h0 = autograd.Variable(torch.zeros(
            self.num_layers * (1 + (1 if self.bidirectional else 0)), x.size(0), self.hidden_size,
            requires_grad=True)).to(self.device)

        c0 = autograd.Variable(torch.zeros(
            self.num_layers * (1 + (1 if self.bidirectional else 0)), x.size(0), self.hidden_size,
            requires_grad=True)).to(self.device)

        if self.type_rnn == 'lstm':
            output, (h, c) = self.body_rnn[0](x, (h0, c0))
        elif self.type_rnn == 'gru':
            output, h = self.body_rnn[0](x, h0)

        h = torch.transpose(h, 0, 1).reshape(x.shape[0], -1, self.hidden_size * (1 + (1 if self.bidirectional else 0)))
        attn_output, attention = self.attention(output, h)
        if self.conv_layers:
            out_conv = self.body_conv(torch.transpose(x, 1, 2)).view(x.shape[0], -1)

            out_for_fc = torch.cat((attn_output.view(x.shape[0], -1), h.reshape(x.shape[0], -1),
                                    output[:, -1:, :].reshape(x.shape[0], -1), out_conv), 1)
        else:
            out_for_fc = torch.cat((attn_output.view(x.shape[0], -1), h.reshape(x.shape[0], -1),
                                    output[:, -1:, :].reshape(x.shape[0], -1)), 1)
        outs = []
        for i in range(self.heads):
            outs.append(self.multi_head[i](out_for_fc))

        out = torch.cat(outs, axis=1).float().to(self.device)
        return out


def get_scaler(path_scaler):
    if get_file(path_scaler):
        with open(path_scaler, 'rb') as f:
            scaler = pickle.load(file=f)
    else:
        file_id = '1lEVf2yECUB-022q-6O_KHxd_6bsFGO8I'
        destination = './scaler_state'
        download_file_from_google_drive(file_id, destination)
        with open(path_scaler, 'rb') as f:
            scaler = pickle.load(file=f)
    return scaler


def get_model(path_model):
    input_size = INPUT_SIZE
    hidden_size = HIDDEN_SIZE
    num_layers = NUM_LAYERS
    num_classes = NUM_CLASSES

    model = BaselineModelLSTM(num_classes, input_size, hidden_size, num_layers,
                              bidirectional=BID, type_rnn=TYPE_RNN, heads=HEADS,
                              conv_layers=CONV_LAYERS, kernel_conv=KERNEL_CONV)
    model.to(DEVICE)
    if get_file(path_model):
        model.load_state_dict(torch.load(path_model, map_location=torch.device(DEVICE)))
    else:
        file_id = '164Rkvu7Ej3x5C9efL6Dp2ocsNDnaYGsK'
        destination = './model_state'
        download_file_from_google_drive(file_id, destination)
        model.load_state_dict(torch.load(path_model, map_location=torch.device(DEVICE)))

    return model


if __name__ == "__main__":
    pass