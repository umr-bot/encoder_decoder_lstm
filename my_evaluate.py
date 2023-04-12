import os
import numpy as np

np.random.seed(1234)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from utils import CharacterTable, transform
from utils import batch, datagen, decode_sequences, decode_sequences_attention
from utils import read_text, tokenize
from utils import restore_model, restore_model_attention
#from model import seq2seq
from model_attention import seq2seq

from utils import transform2, get_type_lists
import argparse

parser = argparse.ArgumentParser(description="Evaluate encoder decoder model on data.")
parser.add_argument("--train_dir", help="path to train directory")
parser.add_argument("--test_dir", help="path to test directory")
parser.add_argument("--model_dir", help="path to model directory")
parser.add_argument("--model_version", help="version of model used, also mathes the number of epochs model trained upto", default=99)
parser.add_argument("--hidden_size", help="hidden layer dimension/size",default=512)

parser.add_argument("--train_batch_size", help="size of batches used in train",default=128)
parser.add_argument("--test_batch_size", help="size of batches used in test",default=256)
parser.add_argument("--sample_mode", help="type of sampling used by the decoder model during inference",default="argmax")
parser.add_argument("--reverse", help="flag whether to reverse spelling of decoder tokens", default=True)
parser.add_argument("--use_attention", help="flag whether to evaluate using attention")

args = parser.parse_args()

#hidden_size = 512
#train_batch_size = 128
#test_batch_size = 256
#sample_mode = 'argmax'
# Input sequences may optionally be reversed,
# shown to increase performance by introducing
# shorter term dependencies between source and target:
# "Learning to Execute"
# http://arxiv.org/abs/1410.4615
# "Sequence to Sequence Learning with Neural Networks"
# https://arxiv.org/abs/1409.3215
#reverse = True

#if __name__ == '__main__':
# extract training tokens
with open(args.train_dir) as f:
    train_tups = [line.strip('\n').split(',') for line in f]
train_dec_tokens, train_tokens = zip(*train_tups)

# extract validation tokens
with open(args.test_dir) as f:
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

train_steps = len(train_tokens) // args.train_batch_size
test_steps = len(test_tokens) // args.test_batch_size
print("Number of train_steps:",train_steps)
print("Number of val_steps:",test_steps)

# Compile the model.
#model, encoder_model, decoder_model = seq2seq(args.hidden_size, nb_input_chars, nb_target_chars)
if args.use_attention=='T': model, encoder_model, decoder_model = restore_model_attention(args.model_dir+"seq2seq_epoch_"+str(args.model_version)+".h5",args.hidden_size)
else: model, encoder_model, decoder_model = restore_model(args.model_dir+"seq2seq_epoch_"+str(args.model_version)+".h5",args.hidden_size)
print(model.summary())

maxlen = max([len(token) for token in train_tokens]) + 2
batch_size=10
for batch in range(0, len(test_tokens), batch_size):
    print('Main Batch {:d}/{:d}'.format(batch + batch_size, len(test_tokens)),end='\r')
    test_encoder, test_decoder, test_target = transform2( test_tokens[batch:batch+batch_size], maxlen, shuffle=False, dec_tokens=test_dec_tokens[batch: batch+batch_size])
    # For every batch - decode a batch of misspelled tokens from the test set
    nb_tokens = len(test_encoder)
    if args.use_attention=='T':
        input_tokens, target_tokens, decoded_tokens = decode_sequences_attention(
            test_encoder, test_target, input_ctable, target_ctable,
            maxlen, args.reverse, encoder_model, decoder_model, nb_tokens,
            sample_mode=args.sample_mode)

    else: input_tokens, target_tokens, decoded_tokens = decode_sequences(
        test_encoder, test_target, input_ctable, target_ctable,
        maxlen, args.reverse, encoder_model, decoder_model, nb_tokens,
        sample_mode=args.sample_mode, random=False)

    with open(args.model_dir+"/model"+str(args.model_version)+"/encoder_tokens",'a') as f:
        for tok in input_tokens: f.write(tok+'\n')
    with open(args.model_dir+"/model"+str(args.model_version)+"/decoder_tokens",'a') as f:
        for tok in decoded_tokens: f.write(tok+'\n')
    with open(args.model_dir+"/model"+str(args.model_version)+"/target_tokens",'a') as f:
        for tok in target_tokens: f.write(tok+'\n')

    #print('-')
    #print('Input tokens:  ', input_tokens)
    #print('Decoded tokens:', decoded_tokens)
    #print('Target tokens: ', target_tokens)
    #print('-')
