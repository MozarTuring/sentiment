import time
from vocab_embed import load_embeddings
from torchtext import data
import random
import pandas as pd
import jieba
import ipdb
from utils import *
from get_args import *
from create_model import create_model
from models import *
from visual_attention import *


# ipdb.set_trace()
torch.manual_seed(1)
random.seed(1)

# args = get_args_cnn()
args = get_args_rnn()
USE_GPU, DEVICE = get_cuda(args)
NUM_ELE = 26

split = 'sentence'

dic_data, text_fields, label_field = get_fields('data/'+split+'/',)
embed_file = split + '_embed3'
pre_embedding, vocab_ls, vocab_dic = load_embeddings(args.EMBEDDING_DIM, text_fields, emb_file=embed_file)

change_vocab(text_fields['test'], vocab_dic, vocab_ls)

aset = set(['味道', '不错', '很', '划算'])
# for ii, it in enumerate(dic_data['test']):
#   if aset.issubset(it.text) and len(it.text) < 100:
#     print(ii, it.text)

test_iter, = data.BucketIterator.splits((dic_data['test'], ), batch_sizes=(args.BATCH_SIZE, ),
                                        sort_key=None, repeat=False, device=DEVICE) # lambda x: len(x.text)

DISCARD = [' ', '\xa0', '\u3000', '\u200a', '\u2006', '\u2028']
LABELS = ['-2', '-1', '0', '1']
df = pd.read_csv('data/testsetb.csv', header=0, encoding='utf-8')
ls_senten = [string_process(df.iloc[row, 1]) for row in range(100)]
print(len(df))

if split == 'char':
  ls_check = [[vocab_dic[st.lower()] for st in senten if st not in DISCARD] for senten in ls_senten]
elif split == 'jieba':
  ls_check = []
  t = 0
  for senten in ls_senten:
    t += 1
    if t % 10000 == 0:
      print(t)
    lss = []
    for st in jieba.cut(senten):
      if st not in DISCARD:
        lss.append(str(vocab_dic[st.lower()]))
    ls_check.append(''.join(lss))
elif split == 'hanlp':
  pass

# vocab_check = {str_sen: i for i, str_sen in enumerate(ls_check)}
# ls_sent_len = [len(se) for se in ls_check]
# print('Maximum length of testset is {}\n'.format(max(ls_sent_len)))

if args.model == 'cnn':
  get_filter_sizes(args)
model = create_model(args, vocab_ls, label_field, USE_GPU, DEVICE, NUM_ELE,)
args.TASK_ID = None
out_dir = make_out_dir(args, NUM_ELE)
print('Loading model from {}'.format(out_dir))
model.load_state_dict(torch.load(out_dir + '/best_model.pth'))
print('Complete loading')

model.eval()
for num_iter, batch in enumerate(test_iter):
  sent = batch.text
  input_sent = [vocab_ls[sen] for sen in sent.squeeze(1).tolist()]
  # print(input_sent)
  if len(input_sent) < 80:
    print(input_sent)
    print(len(input_sent))
    # start1 = time.time()
    pred = model(torch.transpose(sent, 0, 1))
    pred_label = (pred[0].data.max(-1)[1], pred[1].data.max(-1)[1])
    # output_words = pred_label[1].squeeze(1).tolist()
    output_words = ['location', 'service', 'price', 'environment', 'dish', 'others']
    input_se = [''.join([vocab_ls[se] for se in sen]) for sen in pred[3]]
    showAttention(input_se, output_words, pred[2].transpose(0, 1)[20:].detach().cpu(), num_iter)
    continue
  else:
    continue
  # print(time.time()-start1)
  ls_pred = pred_label.tolist()
  sent_ls = torch.transpose(sent, 0, 1).tolist()
  unpad_sent_ls = [[sen for sen in sent if sen != 1] for sent in sent_ls]

  for tt, unpad_sent in enumerate(unpad_sent_ls):
    str_unpad = ''.join(map(str, unpad_sent))
    # start2 = time.time()
    if str_unpad not in vocab_check:
      ipdb.set_trace()
    # print(time.time()-start2)
    for i in range(20):
      df.iloc[vocab_check[str_unpad], i+2] = label_field.vocab.itos[ls_pred[i][tt] + 2]
    # print(time.time()-start2)

sys.exit()
for roww in range(len(df)):
  df.iloc[roww, 1] = ''
  for i in range(20):
    if df.iloc[roww, i+2] not in LABELS:
      ipdb.set_trace()

result_path = 'results/' + get_Model_Name(args) + '.csv'
print('Writing result to {}'.format(result_path))
df.to_csv(result_path, encoding='utf_8_sig', index=False)