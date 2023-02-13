from utils import CharacterTable, transform2
from utils import batch, datagen, datagen_triplet, decode_sequences

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

copy_val_tokens,copy_val_dec_tokens=[],[]
#chrs = input_chars.union(target_chars)
for i in range(len(val_tokens)):
    tok,dec_tok = val_tokens[i], val_dec_tokens[i]
    if set(tok).issubset(input_chars) and set(dec_tok).issubset(target_chars):
        copy_val_tokens.append(tok)
        copy_val_dec_tokens.append(dec_tok)
val_tokens, val_dec_tokens = copy_val_tokens,copy_val_dec_tokens

maxlen = max([len(token) for token in train_tokens]) + 2

epoch=0

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

train_loader = datagen(train_encoder_batch, train_decoder_batch, train_target_batch)
val_loader = datagen(val_encoder_batch, val_decoder_batch, val_target_batch)


