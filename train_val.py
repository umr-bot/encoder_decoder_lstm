import os
import numpy as np

np.random.seed(1234)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow
from utils import CharacterTable, transform
from utils import batch, datagen, decode_sequences, decode_sequences_attention
#from utils import read_text, tokenize
from utils import restore_model
from model_attention import seq2seq as seq2seq_attention
from model import seq2seq
#from model_embed import seq2seq
from utils import transform2, get_type_lists

import argparse

parser = argparse.ArgumentParser(description="Encoder decoder training script")
parser.add_argument("--hidden_size",help="hidden layer size")
#parser.add_argument("--lang",help="language being trained on")
parser.add_argument("--dropout", default=0.2, help="dropout regularization rate")
parser.add_argument("--model_num", default=0, help="dropout regularization rate")
parser.add_argument("--num_epochs", help="hidden layer1 size")
parser.add_argument("--data_dir",help="path to unigrams")
parser.add_argument("--foldset_num",help="fold set to use")
parser.add_argument("--use_attention", default='F',help="flag set to True to use Luong attention, 'T' for true and 'F' for false")
parser.add_argument("--train_batch_size", default=128, help="train batch size")
parser.add_argument("--val_batch_size", default=256, help="val batch size")
parser.add_argument("--checkpoints_dir", default="eng_checkpoints", help="directory in which to save checkpoints")
parser.add_argument("--use_bigrams", default="false", help="boolean which selects whether to load unigram or bigram data")
parser.add_argument("--use_trigrams", default="false", help="boolean which selects whether to load trigram data")

args = parser.parse_args()

#error_rate = 0.8
hidden_size,args.num_epochs,args.train_batch_size,args.val_batch_size, args.model_num = int(args.hidden_size), int(args.num_epochs), int(args.train_batch_size), int(args.val_batch_size), int(args.model_num)
sample_mode = 'argmax'
# Input sequences may optionally be reversed,
# shown to increase performance by introducing
# shorter term dependencies between source and target:
# "Learning to Execute"
# http://arxiv.org/abs/1410.4615
# "Sequence to Sequence Learning with Neural Networks"
# https://arxiv.org/abs/1409.3215
reverse = False

# extract training tokens
if args.use_bigrams == "false":
    #train_dec_tokens are targets and train_tokens inputs
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
elif args.use_trigrams == "true":
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

input_chars = set(' '.join(train_tokens) + '*' + '\t') # * and \t are EOS and SOS respectively
target_chars = set(' '.join(train_dec_tokens) + '*' + '\t')
total_chars = input_chars.union(target_chars)
nb_input_chars = len(input_chars)
nb_target_chars = len(target_chars)
nb_total_chars = len(total_chars)

maxlen = max([len(token) for token in train_tokens]) + 2

# Define training and evaluation configuration.
input_ctable  = CharacterTable(input_chars)
target_ctable = CharacterTable(target_chars)
total_ctable = CharacterTable(total_chars)

train_steps = len(train_tokens) // args.train_batch_size
val_steps = len(val_tokens) // args.val_batch_size
print("Number of train_steps:",train_steps)
print("Number of val_steps:",val_steps)

# Compile the model.
if args.model_num == 0:
    if args.use_attention=='T': model, encoder_model, decoder_model = seq2seq_attention(hidden_size, nb_total_chars, nb_total_chars, dropout=float(args.dropout))
    else: 
        model, encoder_model, decoder_model = seq2seq(hidden_size=hidden_size, nb_input_chars=nb_total_chars, nb_target_chars=nb_total_chars, dropout=float(args.dropout))
        #model, encoder_model, decoder_model = seq2seq(hidden_size=hidden_size, nb_input_chars=nb_total_chars, nb_target_chars=nb_total_chars, maxlen=maxlen, embed_output_dim=nb_total_chars, dropout=float(args.dropout))

else: 
    load_dir = args.checkpoints_dir + '/fold' + str(args.foldset_num)+'_hdd_'+args.hidden_size  + "/seq2seq_epoch_"+str(args.model_num)+".h5"
    model, encoder_model, decoder_model = restore_model(load_dir, hidden_size)

print(model.summary())

# Train and evaluate.
for epoch in range(args.model_num,args.num_epochs):
    #print('Main Epoch {:d}/{:d}'.format(epoch + 1, args.num_epochs))
    print(f"Main Epoch {str(epoch+1)}/{str(args.num_epochs)}")
    #train_encoder, train_decoder, train_target = transform(
    #    vocab, maxlen, error_rate=error_rate, shuffle=True)
    # note shuffle false to keep train_tokens and train_dec_tokens in line
    st_ind,et_ind = int(len(train_tokens) * (epoch/args.num_epochs)), int(len(train_tokens)*((epoch+1)/args.num_epochs))
    sv_ind,ev_ind = int(len(val_tokens) * (epoch/args.num_epochs)), int(len(val_tokens)*((epoch+1)/args.num_epochs))

    train_encoder, train_decoder, train_target = transform2( train_tokens[st_ind:et_ind], maxlen, shuffle=False, dec_tokens=train_dec_tokens[st_ind:et_ind])
    
    val_encoder, val_decoder, val_target = transform2( val_tokens[sv_ind:ev_ind], maxlen, shuffle=False, dec_tokens=val_dec_tokens[sv_ind:ev_ind])

    train_encoder_batch = batch(train_encoder, maxlen, total_ctable, args.train_batch_size, reverse)
    train_decoder_batch = batch(train_decoder, maxlen, total_ctable, args.train_batch_size)
    train_target_batch  = batch(train_target, maxlen, total_ctable, args.train_batch_size)    

    val_encoder_batch = batch(val_encoder, maxlen, total_ctable, args.val_batch_size, reverse)
    val_decoder_batch = batch(val_decoder, maxlen, total_ctable, args.val_batch_size)
    val_target_batch  = batch(val_target, maxlen, total_ctable, args.val_batch_size)

    train_loader = datagen(train_encoder_batch,
                           train_decoder_batch, train_target_batch)
    val_loader = datagen(val_encoder_batch,
                         val_decoder_batch, val_target_batch)
    
    #tensorflow.config.run_functions_eagerly(True)
    history = model.fit(train_loader,
                        steps_per_epoch=train_steps,
                        epochs=1, verbose=1,
                        validation_data=val_loader,
                        validation_steps=val_steps)


    # Save the model at end of each epoch.
    model_file = '_'.join(['seq2seq', 'epoch', str(epoch + 1)]) + '.h5'
    #if args.use_attention=='T' :save_dir = 'checkpoints/luong_'+args.lang+'_checkpoints_fold' + str(fold_num)+'_hdd_'+args.hidden_size
    #else: save_dir = "checkpoints/"+args.lang+'_checkpoints_fold' + str(fold_num)+'_hdd_'+args.hidden_size
    save_dir=args.checkpoints_dir + '/fold' + str(args.foldset_num)+'_hdd_'+args.hidden_size 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if (epoch+1) % 10 == 0:
        save_path = os.path.join(save_dir, model_file)
        print('Saving full model to {:s}'.format(save_path))
        model.save(save_path)
    fn = save_dir+"/history.txt"
    #for history in score:
    with open(fn,'a') as f:
        f.write(str(history.history['loss'][0]) + ',')
        #f.write(str(history.history['truncated_loss'][0]) + ',')
        f.write(str(history.history['val_loss'][0]) + ',')
        #f.write(str(history.history['val_truncated_loss'][0]) + ',')
        f.write(str(history.history['accuracy'][0]) + ',')
        #f.write(str(history.history['truncated_acc'][0]) + ',')
        f.write(str(history.history['val_accuracy'][0]) + ',')
        #f.write(str(history.history['val_truncated_acc'][0]) + ',')
        #f.write(str(history.history['precision'][0]) + ',')
        f.write(str(history.history['recall'][0]) + ',')
        f.write(str(history.history['f1_score'][0])+',')
        #f.write(str(history.history['val_precision'][0]) + ',')
        f.write(str(history.history['val_recall'][0]) + ',')
        f.write(str(history.history['val_f1_score'][0]))
       
        f.write('\n')
   
   # On epoch end - decode a batch of misspelled tokens from the
   # validation set to visualize speller performance.
    nb_tokens = 5
    if args.use_attention=='T':
        input_tokens, target_tokens, decoded_tokens = decode_sequences_attention(
            val_encoder, val_target, total_ctable, total_ctable,
            maxlen, reverse, encoder_model, decoder_model, nb_tokens,
            sample_mode=sample_mode)
    else:
        input_tokens, target_tokens, decoded_tokens = decode_sequences(
            val_encoder, val_target, total_ctable, total_ctable,
            maxlen, reverse, encoder_model, decoder_model, nb_tokens,
            sample_mode=sample_mode, random=False)
        print('-')
        print('Input tokens:  ', input_tokens)
        print('Decoded tokens:', decoded_tokens)
        print('Target tokens: ', target_tokens)
        print('-')

