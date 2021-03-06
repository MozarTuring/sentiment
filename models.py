import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMSentiment(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, use_gpu, batch_size, devic, dropout=0.5):
        super(BiLSTMSentiment, self).__init__()
        self.DEVICE = devic
        self.hidden_dim = hidden_dim
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.dropout = dropout
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True)
        self.hidden2label = nn.Linear(hidden_dim*2, label_size)
        # self.hidden = self.init_hidden()

    # def init_hidden(self):
    #     if self.use_gpu:
    #         return (Variable(torch.zeros((2, self.batch_size, self.hidden_dim), device=self.DEVICE)),
    #                 Variable(torch.zeros((2, self.batch_size, self.hidden_dim), device=self.DEVICE)))
    #     else:
    #         return (Variable(torch.zeros(2, self.batch_size, self.hidden_dim)),
    #                 Variable(torch.zeros(2, self.batch_size, self.hidden_dim)))

    def forward(self, sentence):
        x = self.embeddings(sentence)
        lstm_out, self.hidden = self.lstm(torch.transpose(x, 0, 1))
        y = self.hidden2label(lstm_out[-1])
        log_probs = F.log_softmax(y, dim=1)
        return log_probs


class LSTMSentiment(nn.Module):

  def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, use_gpu, batch_size, devic, dropout=0.5, ):
    super(LSTMSentiment, self).__init__()
    self.DEVICE = devic
    self.hidden_dim = hidden_dim
    self.use_gpu = use_gpu
    self.batch_size = batch_size
    self.dropout = dropout
    self.embeddings = nn.Embedding(vocab_size, embedding_dim)
    self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim)
    self.hidden2label = nn.Linear(hidden_dim, label_size)
    # self.hidden = self.init_hidden()

  # def init_hidden(self):
  #   if self.use_gpu:
  #     return (Variable(torch.zeros((1, self.batch_size, self.hidden_dim), device=self.DEVICE)),
  #             Variable(torch.zeros((1, self.batch_size, self.hidden_dim), device=self.DEVICE)))
  #   else:
  #     return (Variable(torch.zeros(1, self.batch_size, self.hidden_dim)),
  #             Variable(torch.zeros(1, self.batch_size, self.hidden_dim)))

  def forward(self, sentence):
    x = self.embeddings(sentence)
    lstm_out, self.hidden = self.lstm(torch.transpose(x, 0, 1))
    y = self.hidden2label(lstm_out[-1])
    log_probs = F.log_softmax(y, dim=1)
    return log_probs


class GRUSentiment(nn.Module):

  def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
    super(GRUSentiment, self).__init__()
    self.n_layers = n_layers
    self.hidden_size = hidden_size
    self.embeddings = embedding

    # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
    #   because our input size is a word embedding with number of features == hidden_size
    self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                      dropout=(0 if n_layers == 1 else dropout))

  def forward(self, input_seq, hidden=None):
    # Convert word indexes to embeddings
    embedded = self.embedding(input_seq)
    print(embedded.shape)
    # Pack padded batch of sequences for RNN module
    # packed = torch.nn.utils.rnn.pack_padded_sequence(torch.transpose(embedded, 0, 1), input_lengths)
    # Forward pass through GRU
    outputs, hidden = self.gru(embedded, hidden)
    # Unpack padding
    outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
    # Sum bidirectional GRU outputs
    outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
    # Return output and final hidden state
    return outputs, hidden

class CNNSentiment(nn.Module):
  def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout):
    super().__init__()

    self.hdim = 100
    self.bns = nn.ModuleList([nn.BatchNorm1d(n_filters) for _ in filter_sizes])
    self.embeddings = nn.Embedding(vocab_size, embedding_dim)
    self.convs = nn.ModuleList(
      [nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs, embedding_dim)) for fs in filter_sizes])
    self.fc1 = nn.Linear(len(filter_sizes) * n_filters, self.hdim)
    self.fc2 = nn.Linear(self.hdim, output_dim)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    # x = x.permute(1, 0)
    embedded = self.embeddings(x)
    embedded = embedded.unsqueeze(1)
    conved = [F.relu(self.bns[i](conv(embedded).squeeze(3))) for i, conv in enumerate(self.convs)]
    pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
    cat = self.dropout(torch.cat(pooled, dim=1))
    h1 = self.fc1(cat)
    h2 = self.fc2(h1)
    return F.log_softmax(h2, dim=1)
