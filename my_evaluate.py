import os
import numpy as np

np.random.seed(1234)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from utils import CharacterTable, transform
from utils import batch, datagen, decode_sequences
from utils import read_text, tokenize
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

#if __name__ == '__main__':
# extract training tokens
with open("eng_data/kfolds/folds/train") as f: train_tokens = [tok for line in f for tok in line.split()]
with open("eng_data/kfolds/norm_folds/train") as f: train_dec_tokens = [tok for line in f for tok in line.split()]

# Convert train word token lists to type lists
train_tokens,train_dec_tokens = get_type_lists(train_tokens,train_dec_tokens)

# extract validation tokens
with open("eng_data/kfolds/folds/fold4") as f: val_tokens = [tok for line in f for tok in line.split()]
with open("eng_data/kfolds/norm_folds/fold4") as f: val_dec_tokens = [tok for line in f for tok in line.split()]
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

# Compile the model.
#model, encoder_model, decoder_model = seq2seq(hidden_size, nb_input_chars, nb_target_chars)
model_cnt=32
model, encoder_model, decoder_model = restore_model("checkpoints_eng_3_word_types/seq2seq_epoch_"+str(model_cnt)+".h5",hidden_size)
print(model.summary())

maxlen = max([len(token) for token in train_tokens]) + 2
nb_batches = 50 # TODO: update this 
# Train and evaluate. 
#for batch in range(0,nb_batches):
#    print('Bactch {:d}/{:d}'.format(epoch + 1, nb_batches))

s_ind,e_ind=0,20
val_encoder, val_decoder, val_target = transform2( val_tokens[s_ind:e_ind], maxlen, shuffle=False, dec_tokens=val_dec_tokens[s_ind:e_ind])

# For every batch - decode a batch of misspelled tokens from the test set
nb_tokens = 10
input_tokens, target_tokens, decoded_tokens = decode_sequences(
    val_encoder, val_target, input_ctable, target_ctable,
    maxlen, reverse, encoder_model, decoder_model, nb_tokens,
    sample_mode=sample_mode, random=False)

print('-')
print('Input tokens:  ', input_tokens)
print('Decoded tokens:', decoded_tokens)
print('Target tokens: ', target_tokens)
print('-')
