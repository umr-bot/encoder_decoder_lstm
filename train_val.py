import os
import numpy as np

np.random.seed(1234)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow
from utils import CharacterTable, transform
from utils import batch, datagen, decode_sequences
#from utils import read_text, tokenize
from utils import restore_model
from model import seq2seq

from utils import transform2, get_type_lists

#error_rate = 0.8
hidden_size = 512
nb_epochs = 100
train_batch_size = 128
val_batch_size = 256
sample_mode = 'argmax'
# Input sequences may optionally be reversed,
# shown to increase performance by introducing
# shorter term dependencies between source and target:
# "Learning to Execute"
# http://arxiv.org/abs/1410.4615
# "Sequence to Sequence Learning with Neural Networks"
# https://arxiv.org/abs/1409.3215
reverse = True

# extract training tokens
# train_dec_tokens are targets and train_tokens inputs
with open("eng_data_common_misspellings/folds/enc_dec_folds/enc_dec_fold1/train") as f:
    train_tups = [line.strip('\n').split(',') for line in f]
train_dec_tokens, train_tokens = zip(*train_tups)

# Convert train word token lists to type lists
#train_tokens,train_dec_tokens = get_type_lists(train_tokens,train_dec_tokens)

# extract validation tokens
# val_dec_tokens are targets and val_tokens inputs
with open("eng_data_common_misspellings/folds/enc_dec_folds/enc_dec_fold1/val") as f:
    val_tups = [line.strip('\n').split(',') for line in f]
val_dec_tokens, val_tokens = zip(*val_tups)

input_chars = set(' '.join(train_tokens) + '*' + '\t') # * and \t are EOS and SOS respectively
target_chars = set(' '.join(train_dec_tokens) + '*' + '\t')
nb_input_chars = len(input_chars)
nb_target_chars = len(target_chars)
# Define training and evaluation configuration.
input_ctable  = CharacterTable(input_chars)
target_ctable = CharacterTable(target_chars)

train_steps = len(train_tokens) // train_batch_size
val_steps = len(val_tokens) // val_batch_size
print("Number of train_steps:",train_steps)
print("Number of val_steps:",val_steps)

copy_val_tokens,copy_val_dec_tokens=[],[]
#chrs = input_chars.union(target_chars)
for i in range(len(val_tokens)):
    tok,dec_tok = val_tokens[i], val_dec_tokens[i]
    if set(tok).issubset(input_chars) and set(dec_tok).issubset(target_chars):
        copy_val_tokens.append(tok)
        copy_val_dec_tokens.append(dec_tok)
val_tokens, val_dec_tokens = copy_val_tokens,copy_val_dec_tokens

# Compile the model.
model, encoder_model, decoder_model = seq2seq(hidden_size, nb_input_chars, nb_target_chars)
model_cnt=0
#model, encoder_model, decoder_model = restore_model("checkpoints/seq2seq_epoch_"+str(model_cnt)+".h5",hidden_size)
print(model.summary())

maxlen = max([len(token) for token in train_tokens]) + 2

# Train and evaluate.
for epoch in range(model_cnt,100):
    print('Main Epoch {:d}/{:d}'.format(epoch + 1, nb_epochs))

    #train_encoder, train_decoder, train_target = transform(
    #    vocab, maxlen, error_rate=error_rate, shuffle=True)
    # note shuffle false to keep train_tokens and train_dec_tokens in line
    st_ind,et_ind = int(len(train_tokens) * (epoch/100)), int(len(train_tokens)*((epoch+1)/100))
    sv_ind,ev_ind = int(len(val_tokens) * (epoch/100)), int(len(val_tokens)*((epoch+1)/100))

    train_encoder, train_decoder, train_target = transform2( train_tokens[st_ind:et_ind], maxlen, shuffle=False, dec_tokens=train_dec_tokens[st_ind:et_ind])
    
    val_encoder, val_decoder, val_target = transform2( val_tokens[sv_ind:ev_ind], maxlen, shuffle=False, dec_tokens=val_dec_tokens[sv_ind:ev_ind])

    train_encoder_batch = batch(train_encoder, maxlen, input_ctable, train_batch_size, reverse)
    train_decoder_batch = batch(train_decoder, maxlen, target_ctable, train_batch_size)
    train_target_batch  = batch(train_target, maxlen, target_ctable, train_batch_size)    

    val_encoder_batch = batch(val_encoder, maxlen, input_ctable, val_batch_size, reverse)
    val_decoder_batch = batch(val_decoder, maxlen, target_ctable, val_batch_size)
    val_target_batch  = batch(val_target, maxlen, target_ctable, val_batch_size)

    train_loader = datagen(train_encoder_batch,
                           train_decoder_batch, train_target_batch)
    val_loader = datagen(val_encoder_batch,
                         val_decoder_batch, val_target_batch)
    tensorflow.config.run_functions_eagerly(True)
    history = model.fit(train_loader,
                        steps_per_epoch=train_steps,
                        epochs=1, verbose=1,
                        validation_data=val_loader,
                        validation_steps=val_steps)

   # On epoch end - decode a batch of misspelled tokens from the
   # validation set to visualize speller performance.
    nb_tokens = 5
    input_tokens, target_tokens, decoded_tokens = decode_sequences(
        val_encoder, val_target, input_ctable, target_ctable,
        maxlen, reverse, encoder_model, decoder_model, nb_tokens,
        sample_mode=sample_mode, random=False)
    
    print('-')
    print('Input tokens:  ', input_tokens)
    print('Decoded tokens:', decoded_tokens)
    print('Target tokens: ', target_tokens)
    print('-')

    # Save the model at end of each epoch.
    model_file = '_'.join(['seq2seq', 'epoch', str(epoch + 1)]) + '.h5'
    save_dir = 'checkpoints'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, model_file)
    print('Saving full model to {:s}'.format(save_path))
    model.save(save_path)
    fn = 'history_with_trunc.txt'
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

