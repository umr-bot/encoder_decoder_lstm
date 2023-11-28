# coding: utf-8
import os
# NOTE: log levels for tensorflow
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import Model, load_model
from tqdm import tqdm
from model import precision,recall,f1_score
from utils import CharacterTable, transform2, decode_sequences, restore_model
import argparse

parser = argparse.ArgumentParser(description="Logistic regression training script")
parser.add_argument("--hidden_size", default=128, help="hidden layer1 size")
#parser.add_argument("--train_batch_size", default=10, help="train batch size")
#parser.add_argument("--val_batch_size", default=100, help="val batch size")
#parser.add_argument("--num_epochs", default=20, help="number of epochs")
parser.add_argument("--foldset_num", default=1, help="foldset number to use")
parser.add_argument("--checkpoints_dir", default="eng_checkpoints", help="directory in which to save checkpoints")
parser.add_argument("--data_dir",help="path to unigrams")
#parser.add_argument("--lang",help="language being trained on")
parser.add_argument("--model_num",help="which model version is being evaluated on")
parser.add_argument("--use_bigrams", default="false", help="boolean which selects whether to load unigram or bigram data")
parser.add_argument("--use_trigrams", default="false", help="boolean which selects whether to load trigram data")

args = parser.parse_args()
args.hidden_size=int(args.hidden_size)
#assert(len(int(args.val_batch_size)) < len(int(args.)))

sample_mode = 'argmax'

# Load data
if args.use_bigrams=="false":
    with open(args.data_dir+"/foldset"+str(args.foldset_num)+"/train") as f:
        train_tups = [line.strip('\n').split(',') for line in f]
    train_dec_tokens, train_tokens = zip(*train_tups)
    with open(args.data_dir+"/foldset"+str(args.foldset_num)+"/val") as f:
        val_tups = [line.strip('\n').split(',') for line in f]
    val_dec_tokens, val_tokens = zip(*val_tups)
elif args.use_bigrams == "right" or args.use_bigrams == "left":
    # selects bigram (n=2) or trigram (n=3)
    if args.use_bigrams == "left": m,n=0,2 # left bigram (m=0,n=2)
    else: m,n=-2,3 # right bigram (m=-2,n=3)
    with open(args.data_dir+"/foldset"+str(args.foldset_num)+"/train") as f:
        train_tokens = [" ".join(line.strip('\n').split()[m:n]) for line in f]
    with open(args.data_dir+"/norm_foldset"+str(args.foldset_num)+"/train") as f:
        train_dec_tokens = [" ".join(line.strip('\n').split()[m:n]) for line in f]

    with open(args.data_dir+"/foldset"+str(args.foldset_num)+"/val") as f:
        val_tokens = [" ".join(line.strip('\n').split()[m:n]) for line in f]
    with open(args.data_dir+"/norm_foldset"+str(args.foldset_num)+"/val") as f:
        val_dec_tokens = [" ".join(line.strip('\n').split()[m:n]) for line in f]

elif args.use_trigrams == "True":
    # selects bigram (n=2) or trigram (n=3)
    n=3
    with open(args.data_dir+"/foldset"+str(args.foldset_num)+"/train") as f:
        train_tokens = [" ".join(line.strip('\n').split()[:n]) for line in f]
    with open(args.data_dir+"/norm_foldset"+str(args.foldset_num)+"/train") as f:
        train_dec_tokens = [" ".join(line.strip('\n').split()[:n]) for line in f]

    with open(args.data_dir+"/foldset"+str(args.foldset_num)+"/val") as f:
        val_tokens = [" ".join(line.strip('\n').split()[:n]) for line in f]
    with open(args.data_dir+"/norm_foldset"+str(args.foldset_num)+"/val") as f:
        val_dec_tokens = [" ".join(line.strip('\n').split()[:n]) for line in f]

else: print("ERROR: data not loaded")

#    with open(+"/foldset"+str(args.foldset_num)+"/val") as f:
#        val_tups = [line.strip('\n').split(',') for line in f]
#    val_dec_tokens, val_tokens = zip(*val_tups)

input_chars = set(' '.join(train_tokens) + '*' + '\t') # * and \t are EOS and SOS respectively
target_chars = set(' '.join(train_dec_tokens) + '*' + '\t')
total_chars = input_chars.union(target_chars)
nb_total_chars = len(total_chars)
#input_ctable = CharacterTable(total_chars)
#target_ctable = CharacterTable(target_chars)
total_ctable = input_ctable = target_ctable = CharacterTable(total_chars)
maxlen = max([len(token) for token in train_tokens]) + 2
#########################################################################################

# Load model
custom_objects = { 'recall': recall, "precision": precision, "f1_score": f1_score}
root_dir=args.checkpoints_dir+'/fold' + str(args.foldset_num)+'_hdd_'+str(args.hidden_size)
#model, encoder_model, decoder_model = load_model( root_dir + "/seq2seq_epoch_"+args.model_num+".h5",custom_objects=custom_objects)
model, encoder_model, decoder_model = restore_model(root_dir + "/seq2seq_epoch_"+args.model_num+".h5", args.hidden_size, compile_flag=False)

#########################################################################################

val_x, val_y_pred,val_y_true=[],[],[]
val_x_padded, val_y_padded, val_y_true_padded = transform2( val_tokens, maxlen, shuffle=False, dec_tokens=val_dec_tokens)
# when batch_size=1 loads 1 sample at a time
#val_X_iter = batch(val_x_padded,maxlen=maxlen,ctable=total_ctable,batch_size=batch_size,reverse=False)
#val_y_iter = batch(val_y_padded,maxlen=maxlen,ctable=total_ctable,batch_size=batch_size,reverse=False)
#val_y_true_iter = batch(val_y_true_padded,maxlen=maxlen,ctable=total_ctable,batch_size=batch_size,reverse=False)

#val_loader = datagen(val_X_iter, val_y_iter, val_y_true_iter)

batch_size=10
for batch_cnt in tqdm(range(0, len(val_tokens)-batch_size, batch_size),desc="Inference in progress"):
    val_x_temp, val_y_true_temp, val_y_pred_temp = decode_sequences(
        val_x_padded[batch_cnt:batch_cnt+batch_size], val_y_true_padded[batch_cnt:batch_cnt+batch_size], input_ctable=total_ctable, target_ctable=total_ctable,
        maxlen=maxlen, encoder_model=encoder_model, decoder_model=decoder_model, nb_examples=batch_size,
        sample_mode=sample_mode,reverse=False, random=False)
    val_x += val_x_temp
    val_y_pred += val_y_pred_temp
    val_y_true += val_y_true_temp

save_dir = root_dir+"/model_"+args.model_num
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
with open(save_dir+"/err_file",'w') as f:
    for word in val_x: f.write(word+'\n')
with open(save_dir+"/cln_file",'w')as f:
    for word in val_y_pred: f.write(word+'\n')
with open(save_dir+"/tar_file",'w') as f:
    for word in val_y_true: f.write(word+'\n')

