import models
import models2
import torch.nn as nn


def create_model(args, vocab_ls, label_field, USE_GPU, DEVICE):
  if args.model == 'lstm':
    print('Creating %s model' % args.model)
    model = models.LSTMSentiment(embedding_dim=args.EMBEDDING_DIM, hidden_dim=args.HIDDEN_DIM,
                                 vocab_size=len(vocab_ls), label_size=len(label_field.vocab.itos) - 1,
                                 use_gpu=USE_GPU, batch_size=args.BATCH_SIZE, devic=DEVICE)

  if args.model == 'lstm_attention':
    print('Creating %s model' % args.model)
    model = models2.LSTMSentiment(embedding_dim=args.EMBEDDING_DIM, hidden_dim=args.HIDDEN_DIM,
                                  vocab_size=len(vocab_ls.vocab), label_size=len(label_field.vocab.itos) - 1,
                                  use_gpu=USE_GPU, batch_size=args.BATCH_SIZE, devic=DEVICE)

  if args.model == 'bilstm':
    print('Creating %s model' % args.model)
    model = models.BiLSTMSentiment(embedding_dim=args.EMBEDDING_DIM, hidden_dim=args.HIDDEN_DIM,
                                   vocab_size=len(vocab_ls), label_size=len(label_field.vocab.itos) - 1,
                                   use_gpu=USE_GPU, batch_size=args.BATCH_SIZE, devic=DEVICE)

  if args.model == 'bilstm_attention':
    print('Creating %s model' % args.model)
    model = models2.BiLSTMSentiment(embedding_dim=args.EMBEDDING_DIM, hidden_dim=args.HIDDEN_DIM,
                                    vocab_size=len(vocab_ls), label_size=len(label_field.vocab.itos) - 1,
                                    use_gpu=USE_GPU, batch_size=args.BATCH_SIZE, devic=DEVICE, attn_method=args.attn_method)

  if args.model == 'gru':
    print('Creating %s model' % args.model)
    embedding = nn.Embedding(len(vocab_ls), args.EMBEDDING_DIM)
    model = models.GRUSentiment(hidden_size=args.HIDDEN_DIM, embedding=embedding, n_layers=1, dropout=0)

  if args.model == 'gru_attention':
    print('Creating %s model' % args.model)
    embedding = nn.Embedding(len(vocab_ls), args.EMBEDDING_DIM)
    model = models2.GRUSentiment(hidden_size=args.HIDDEN_DIM, embedding=embedding, n_layers=1, dropout=0)

  if args.paral:
    model = nn.DataParallel(model)

  model.to(DEVICE)

  return model