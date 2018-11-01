from torch import optim
import random
import os
from tqdm import tqdm
from models import *
from torchtext import data
import argparse
from utils import get_accuracy
from utils import load_review_train
from utils import evaluate


# torch.set_num_threads(8)
torch.manual_seed(1)
random.seed(1)


args = argparse.ArgumentParser()
args.add_argument('--m', dest='model', default='bilstm', help='specify the mode to use (default: lstm)')
args.add_argument('--p', dest='paral', default=False, type=bool)
args.add_argument('--id', dest='taskid', type=str)
args = args.parse_args()

TASKID = args.taskid

# cuda_str = "cuda:" + str(int(TASKID) % 8)
cuda_str = 'cuda:0'
DEVICE = torch.device(cuda_str if torch.cuda.is_available() else "cpu")


def train_epoch_progress(model, train_iter, loss_function, optimizer, epoch):
    model.train()
    avg_loss = 0.0
    truth_res = []
    pred_res = []
    count = 0
    for batch in train_iter:
        sent, label = batch.text, batch.label
        label.data.sub_(1)
        truth_res += list(label.data)
        model.batch_size = len(label.data)
        pred = model(torch.transpose(sent, 0, 1))
        pred_label = pred.data.max(1)[1]
        pred_res += [x for x in pred_label]
        model.zero_grad()
        loss = loss_function(pred, label)
        avg_loss += loss.data
        count += 1
        loss.backward()
        optimizer.step()
    avg_loss /= len(train_iter)
    acc = get_accuracy(truth_res, pred_res)
    return avg_loss, acc


EPOCHS = 20
USE_GPU = torch.cuda.is_available()
EMBEDDING_DIM = 200
HIDDEN_DIM = 200

BATCH_SIZE = 32
best_dev_acc = 0.0

text_field = data.Field(lower=True)
label_field = data.Field(sequential=False)
train_iter, dev_iter, test_iter = load_review_train(text_field, label_field, BATCH_SIZE, TASKID, DEVICE)

if args.model == 'lstm':
  print('Creating %s model' % args.model)
  model = LSTMSentiment(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM,
                      vocab_size=len(text_field.vocab), label_size=len(label_field.vocab)-1,
                      use_gpu=USE_GPU, batch_size=BATCH_SIZE, devic=DEVICE)

if args.model == 'bilstm':
  print('Creating %s model' % args.model)
  model = BiLSTMSentiment(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM,
                          vocab_size=len(text_field.vocab), label_size=len(label_field.vocab)-1,
                          use_gpu=USE_GPU, batch_size=BATCH_SIZE, devic=DEVICE)

if args.model == 'gru':
  print('Creating %s model' % args.model)
  embedding = nn.Embedding(len(text_field.vocab), EMBEDDING_DIM)
  model = GRUSentiment(hidden_size=HIDDEN_DIM, embedding=embedding, n_layers=1, dropout=0)

if args.paral:
  model = nn.DataParallel(model)

model.to(DEVICE)

best_model = model
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_function = nn.NLLLoss()

print('Training...')
out_dir = os.path.abspath(os.path.join(os.path.curdir, "task"+TASKID+'/'+args.model))
print("Writing to {}\n".format(out_dir))
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

for epoch in range(EPOCHS):
    avg_loss, acc = train_epoch_progress(model, train_iter, loss_function, optimizer, epoch)
    print('Train: loss %.2f acc %.1f' % (avg_loss, acc*100))
    dev_acc = evaluate(model, dev_iter, loss_function, 'Dev')
    if dev_acc > best_dev_acc:
        if best_dev_acc > 0:
            os.system('rm ' + out_dir + '/best_model' + '.pth')
        best_dev_acc = dev_acc
        best_model = model
        torch.save(best_model.state_dict(), out_dir + '/best_model' + '.pth')
        # evaluate on test with the best dev performance model
        test_acc = evaluate(best_model, test_iter, loss_function, 'Test')

test_acc = evaluate(best_model, test_iter, loss_function, 'Final Test')

