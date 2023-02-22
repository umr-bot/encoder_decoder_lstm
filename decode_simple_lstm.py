from tensorflow.keras.models import Model, load_model
from model import recall, f1_score
from utils import transform2, CharacterTable, batch
import numpy as np
import re

SOS = '\t' # start of sequence.
EOS = '*' # end of sequence.

with open("eng_data_common_misspellings/folds/enc_dec_folds/enc_dec_fold1/train") as f:
    train_tups = [line.strip('\n').split(',') for line in f]
train_dec_tokens, train_tokens = zip(*train_tups)

with open("eng_data_common_misspellings/folds/enc_dec_folds/enc_dec_fold1/train") as f:
    train_tups = [line.strip('\n').split(',') for line in f]
train_dec_tokens, train_tokens = zip(*train_tups)

with open("eng_data_common_misspellings/folds/enc_dec_folds/enc_dec_fold1/train") as f:
    test_tups = [line.strip('\n').split(',') for line in f]
test_dec_tokens, test_tokens = zip(*test_tups)

with open("eng_data_common_misspellings/folds/enc_dec_folds/enc_dec_fold1/train") as f:
    test_tups = [line.strip('\n').split(',') for line in f]
test_dec_tokens, test_tokens = zip(*test_tups)

input_chars = set(' '.join(train_tokens) + '*' + '\t') # * and \t are EOS and SOS respectively
nb_input_chars = len(input_chars)


maxlen = max([len(token) for token in test_tokens]) + 2

model = load_model("simple_lstm_checkpoints/lstm_model_epoch_98.h5", custom_objects={'recall': recall, "f1_score": f1_score})
start_ind,end_ind=20,30

test_encoder, test_decoder, test_target = transform2( test_tokens[start_ind:end_ind], maxlen, shuffle=False, dec_tokens=test_dec_tokens[start_ind:end_ind])

inputs = test_encoder 
targets = test_target 
ctable  = CharacterTable(input_chars)

nb_examples=10
sample_mode='argmax'
reverse=False
input_tokens = []

indices = range(nb_examples)
for index in indices:
    input_tokens.append(inputs[index])
input_sequences = batch(input_tokens, maxlen, ctable, nb_examples, reverse)
input_sequences = next(input_sequences)

# Encode the input as state vectors.    
states_value = model.predict(input_sequences)

# Create batch of empty target sequences of length 1 character.
target_sequences = np.zeros((nb_examples, 1, ctable.size))
# Populate the first element of target sequence
# with the start-of-sequence character.
target_sequences[:, 0, ctable.char2index[SOS]] = 1.0

# Sampling loop for a batch of sequences.
# Exit condition: either hit max character limit
# or encounter end-of-sequence character.
decoded_tokens = [''] * nb_examples
for _ in range(maxlen):
    # `char_probs` has shape: (nb_examples, 1, nb_target_chars)
    char_probs = model.predict([target_sequences])

    # Reset the target sequences.
    target_sequences = np.zeros((nb_examples, 1, ctable.size))

    # Sample next character using argmax or multinomial mode.
    sampled_chars = []
    for i in range(nb_examples):
        if sample_mode == 'argmax':
            next_index, next_char = ctable.decode(char_probs[i], calc_argmax=True)
        elif sample_mode == 'multinomial':
            next_index, next_char = ctable.sample_multinomial(char_probs[i], temperature=0.5)
        else: raise Exception("`sample_mode` accepts `argmax` or `multinomial`.")

        decoded_tokens[i] += next_char
        sampled_chars.append(next_char)
        # Update target sequence with index of next character.
        target_sequences[i, 0, next_index] = 1.0

    stop_char = set(sampled_chars)
    if len(stop_char) == 1 and stop_char.pop() == EOS:
        break

input_tokens   = [re.sub('[%s]' % EOS, '', token)
                  for token in input_tokens]
decoded_tokens = [re.sub('[%s]' % EOS, '', token)
                  for token in decoded_tokens]

