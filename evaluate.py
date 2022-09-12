import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from utils import CharacterTable, transform
from utils import restore_model, decode_sequences
from utils import read_text, tokenize
from utils import transform2

#error_rate = 0.6
reverse = True
model_path = './checkpoints_eng_3_word_types/seq2seq_epoch_32.h5'
#model_path = './checkpoints/seq2seq_epoch_32.h5'
hidden_size = 512
sample_mode = 'argmax'
#data_path = './data'
#books = ['nietzsche.txt', 'pride_and_prejudice.txt', 'shakespeare.txt', 'war_and_peace.txt']

#test_sentence = 'The rabbit-hole went straight on like a tunnel for some way, and then dipped suddenly down, so suddenly that Alice had not a moment to think about stopping herself before she found herself falling down a very deep well.'


#if __name__ == '__main__':
#text  = read_text(data_path, books)
#vocab = tokenize(text)
#vocab = list(filter(None, set(vocab)))
## `maxlen` is the length of the longest word in the vocabulary
## plus two SOS and EOS characters.
#maxlen = max([len(token) for token in vocab]) + 2
#train_encoder, train_decoder, train_target = transform(
#    vocab, maxlen, error_rate=error_rate, shuffle=False)
#
#tokens = tokenize(test_sentence)
#tokens = list(filter(None, tokens))
#nb_tokens = len(tokens)
#misspelled_tokens, _, target_tokens = transform(
#    tokens, maxlen, error_rate=error_rate, shuffle=False)
#
#input_chars = set(' '.join(train_encoder))
#target_chars = set(' '.join(train_decoder))
#input_ctable = CharacterTable(input_chars)
#target_ctable = CharacterTable(target_chars)
#base_dir = "data/folds/"
base_dir = "eng_data/kfolds/folds/"
#norm_base_dir = "data/norm_folds/"
norm_base_dir = "eng_data/kfolds/norm_folds/"

# extract training tokens
with open(base_dir + "train") as f: train_tokens = [tok for line in f for tok in line.split()]
with open(norm_base_dir + "train") as f: train_dec_tokens = [tok for line in f for tok in line.split()]
# `maxlen` is the length of the longest word in the vocabulary
# plus two SOS and EOS characters.
maxlen = max([len(token) for token in train_tokens]) + 2
#transform2(train_tokens, maxlen, shuffle=False, dec_tokens=train_dec_tokens)

# extract test tokens
with open(base_dir+"fold5") as f: test_tokens = [tok for line in f for tok in line.split()]
with open(norm_base_dir+"fold5") as f: test_target_tokens = [tok for line in f for tok in line.split()]
#transform2(test_tokens, maxlen, shuffle=False, dec_tokens=test_target_tokens)

#tokens = tokenize(test_sentence)
#tokens = list(filter(None, tokens))
#nb_tokens = len(tokens)
#misspelled_tokens, _, target_tokens = transform(
#    tokens, maxlen, error_rate=error_rate, shuffle=False)
assert(len(test_tokens) == len(test_target_tokens))

input_chars = set(' '.join(train_tokens) + '*' + '\t') # * and \t are EOS and SOS respectively
target_chars = set(' '.join(train_dec_tokens) + '*' + '\t')
nb_input_chars = len(input_chars)
nb_target_chars = len(target_chars)
# Define training and evaluation configuration.
input_ctable  = CharacterTable(input_chars)
target_ctable = CharacterTable(target_chars)

_ , encoder_model, decoder_model = restore_model(model_path, hidden_size)
start_index = 10
nb_tokens = 20#len(test_tokens)

input_tokens, target_tokens, decoded_tokens = decode_sequences(
        test_tokens[start_index:nb_tokens], test_target_tokens[start_index:nb_tokens], input_ctable,
        target_ctable, 
        maxlen, reverse, encoder_model, decoder_model, nb_tokens-start_index,
        sample_mode=sample_mode, random=False)

print('-')
print('Input sentence:  ', ' '.join([token for token in input_tokens]))
print('-')
print('Decoded sentence:', ' '.join([token for token in decoded_tokens]))
print('-')
print('Target sentence: ', ' '.join([token for token in target_tokens]))
