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
test_batch_size = 256
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
with open("eng_data_common_misspellings/folds/enc_dec_folds/enc_dec_fold1/train") as f:
    train_tups = [line.strip('\n').split(',') for line in f]
train_dec_tokens, train_tokens = zip(*train_tups)

# extract validation tokens
with open("eng_data_common_misspellings/folds/enc_dec_folds/enc_dec_fold1/test") as f:
    test_tups = [line.strip('\n').split(',') for line in f]
test_dec_tokens, test_tokens = zip(*test_tups)

input_chars = set(' '.join(train_tokens) + '*' + '\t' +'a') # * and \t are EOS and SOS respectively
target_chars = set(' '.join(train_dec_tokens) + '*' + '\t'+'a')
# The exxclamation ! is used for out of vocaulary chars in test set
nb_input_chars = len(input_chars)
nb_target_chars = len(target_chars)
# Define training and evaluation configuration.
input_ctable  = CharacterTable(input_chars)
target_ctable = CharacterTable(target_chars)

train_steps = len(train_tokens) // train_batch_size
test_steps = len(test_tokens) // test_batch_size
print("Number of train_steps:",train_steps)
print("Number of val_steps:",test_steps)

# Compile the model.
#model, encoder_model, decoder_model = seq2seq(hidden_size, nb_input_chars, nb_target_chars)
model_cnt=99
model, encoder_model, decoder_model = restore_model("checkpoints/seq2seq_epoch_"+str(model_cnt)+".h5",hidden_size)
print(model.summary())

maxlen = max([len(token) for token in train_tokens]) + 2
batch_size=10
for batch in range(0, len(test_tokens), batch_size):
    print('Main Batch {:d}/{:d}'.format(batch + batch_size, len(test_tokens)))
    test_encoder, test_decoder, test_target = transform2( test_tokens[batch:batch+batch_size], maxlen, shuffle=False, dec_tokens=test_dec_tokens[batch: batch+batch_size])
    # For every batch - decode a batch of misspelled tokens from the test set
    nb_tokens = len(test_encoder)
    input_tokens, target_tokens, decoded_tokens = decode_sequences(
        test_encoder, test_target, input_ctable, target_ctable,
        maxlen, reverse, encoder_model, decoder_model, nb_tokens,
        sample_mode=sample_mode, random=False)

    with open("checkpoints/model99/encoder_tokens",'a') as f:
        for tok in input_tokens: f.write(tok+'\n')
    with open("checkpoints/model99/decoder_tokens",'a') as f:
        for tok in decoded_tokens: f.write(tok+'\n')
    with open("checkpoints/model99/target_tokens",'a') as f:
        for tok in target_tokens: f.write(tok+'\n')

    #print('-')
    #print('Input tokens:  ', input_tokens)
    #print('Decoded tokens:', decoded_tokens)
    #print('Target tokens: ', target_tokens)
    #print('-')
