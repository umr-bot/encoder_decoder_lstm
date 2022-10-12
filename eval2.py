import os
import numpy as np

np.random.seed(1234)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from utils import CharacterTable, transform
from utils import decode_sequences
from utils import read_text, tokenize
from utils import restore_model
from model import seq2seq

from utils import transform2, get_type_lists
from tqdm import tqdm
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

# extract training tokens
with open("bam_folds/half_combined_folds/train") as f: train_tups = [tup.split(',') for tup in f]
train_tokens,train_dec_tokens = zip(*train_tups)
train_dec_tokens = [tok.strip('\n') for tok in train_dec_tokens]

# Convert train word token lists to type lists
#train_tokens,train_dec_tokens = get_type_lists(train_tokens,train_dec_tokens)

# extract validation tokens
with open("bam_folds/half_combined_folds/test") as f: test_tups = [tup.split(',') for tup in f]
test_tokens,test_dec_tokens = zip(*test_tups)
test_dec_tokens = [tok.strip('\n') for tok in test_dec_tokens]

input_chars = set(' '.join(train_tokens) + '*' + '\t') # * and \t are EOS and SOS respectively
target_chars = set(' '.join(train_dec_tokens) + '*' + '\t')
nb_input_chars = len(input_chars)
nb_target_chars = len(target_chars)
# Define training and evaluation configuration.
input_ctable  = CharacterTable(input_chars)
target_ctable = CharacterTable(target_chars)

copy_test_tokens,copy_test_dec_tokens=[],[]
#chrs = input_chars.union(target_chars)
for i in range(len(test_tokens)):
    tok,dec_tok = test_tokens[i], test_dec_tokens[i]
    if set(tok).issubset(input_chars) and set(dec_tok).issubset(target_chars):
        copy_test_tokens.append(tok)
        copy_test_dec_tokens.append(dec_tok)
test_tokens, test_dec_tokens = copy_test_tokens,copy_test_dec_tokens

# Compile the model.
#model, encoder_model, decoder_model = seq2seq(hidden_size, nb_input_chars, nb_target_chars)
#model_cnt=40
#model, encoder_model, decoder_model = restore_model("checkpoints/seq2seq_epoch_"+str(model_cnt)+".h5",hidden_size, lstm_2_flag=False)
#print(model.summary())
#
#maxlen = max([len(token) for token in train_tokens]) + 2
#
## Evaluate loop.
#start = 0
#batch_size = 10
##for batch in tqdm(range(start,len(test_sentences))):
#for i in tqdm(range(start,len(test_tokens), batch_size)):
#    #if batch%250==0: print(f"[batch {batch}/{len(test_sentences)}]")
#    #test_sentence, test_norm_sentence = test_sentences[batch], test_norm_sentences[batch]
#    #test_tokens = test_sentence.split()
#    #test_dec_tokens = test_norm_sentence.split()
#    test_tokens_batch = test_tokens[i:i+batch_size]
#    test_dec_tokens_batch = test_dec_tokens[i:i+batch_size]
#    test_encoder, test_decoder, test_target = transform2( test_tokens_batch, maxlen, shuffle=False, dec_tokens=test_dec_tokens_batch)
#    if test_encoder == [] :
#        # this case happens when there are only words with lengths less than
#        # three in a test sentence. We only consider word lengths > 3
#        # in the function transform2. Hence words are written as was inputted below.
#        #input_tokens, target_tokens, decoded_tokens=test_tokens,test_dec_tokens,test_dec_tokens
#        continue
#    else: 
#        # Decode a batch of misspelled tokens from the test set
#        nb_tokens = len(test_encoder)
#        input_tokens, target_tokens, decoded_tokens = decode_sequences(
#            test_encoder, test_target, input_ctable, target_ctable,
#            maxlen, reverse, encoder_model, decoder_model, nb_tokens,
#            sample_mode=sample_mode, random=False)
#    
#    with open("toks","a") as f:
#        for input_token,decoded_token,target_token in zip(input_tokens,decoded_tokens,target_tokens):
#            f.write(input_token+',')
#            f.write(decoded_token+',')
#            f.write(target_token)
#            f.write('\n')
#        f.write('\n')
#    print('-')
#    print('Input tokens:  ', input_tokens)
#    print('Decoded tokens:', decoded_tokens)
#    print('Target tokens: ', target_tokens)
#    print('-')
