import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F



class Attn(torch.nn.Module):
  def __init__(self, method, hidden_size):
    super(Attn, self).__init__()
    self.method = method
    if self.method not in ['dot', 'general', 'concat']:
      raise ValueError(self.method, "is not an appropriate attention method.")
    self.hidden_size = hidden_size
    if self.method == 'general':
      self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
    elif self.method == 'concat':
      self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
      self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

  def dot_score(self, hidden, encoder_output):
    return torch.sum(hidden * encoder_output, dim=2)

  def general_score(self, hidden, encoder_output):
    energy = self.attn(encoder_output)
    return torch.sum(hidden * energy, dim=2)

  def concat_score(self, hidden, encoder_output):
    energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
    return torch.sum(self.v * energy, dim=2)

  def forward(self, hidden, encoder_outputs):

    if self.method == 'general':
      attn_energies = self.general_score(hidden, encoder_outputs)
    elif self.method == 'concat':
      attn_energies = self.concat_score(hidden, encoder_outputs)
    elif self.method == 'dot':
      attn_energies = self.dot_score(hidden, encoder_outputs)

    attn_energies = attn_energies.t()

    return F.softmax(attn_energies, dim=1).unsqueeze(1)



class BiLSTMSentiment(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size,
                 label_size, use_gpu, batch_size, devic, attn_method, dropout=0.5):
        super(BiLSTMSentiment, self).__init__()
        self.DEVICE = devic
        self.hidden_dim = hidden_dim
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.dropout = dropout
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True)
        self.hidden2label = nn.Linear(hidden_dim*2, label_size)
        self.attn = Attn(attn_method, hidden_dim*2)
        self.task_vec = Variable(torch.zeros((hidden_dim * 2,), device=devic))

    def forward(self, sentence):
        x = self.embeddings(sentence)
        lstm_out, self.hidden = self.lstm(torch.transpose(x, 0, 1))
        attn_weights = self.attn(self.task_vec, lstm_out)
        context = attn_weights.bmm(lstm_out.transpose(0, 1))
        y = self.hidden2label(context).squeeze(1)
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

    self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                      dropout=(0 if n_layers == 1 else dropout))

  def forward(self, input_seq, hidden=None):
    embedded = self.embedding(input_seq)
    print(embedded.shape)
    outputs, hidden = self.gru(embedded, hidden)
    outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
    outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
    return outputs, hidden