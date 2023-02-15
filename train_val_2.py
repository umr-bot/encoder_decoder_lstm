# coding: utf-8
from tqdm import tqdm
from utils import CharacterTable, transform2, batch_triplet, datagen_triplet

from utils import restore_model
from tri_model import seq2seq

import os
import numpy as np

np.random.seed(1234)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

hidden_size = 80
nb_epochs = 100
train_batch_size = 10
val_batch_size = 20
sample_mode = 'argmax'
# Input sequences may optionally be reversed,
# shown to increase performance by introducing
# shorter term dependencies between source and target:
# "Learning to Execute"
# http://arxiv.org/abs/1410.4615
# "Sequence to Sequence Learning with Neural Networks"
# https://arxiv.org/abs/1409.3215
reverse = True


with open("eng_data_common_misspellings/folds/eng_za_trigrams/fold1/train") as f:
    train = [line.strip('\n').split() for line in f]
with open("eng_data_common_misspellings/folds/eng_za_trigrams/norm_fold1/train") as f:
    norm_train = [line.strip('\n').split() for line in f]

with open("eng_data_common_misspellings/folds/eng_za_trigrams/fold1/val") as f:
    val = [line.strip('\n').split() for line in f]
with open("eng_data_common_misspellings/folds/eng_za_trigrams/norm_fold1/val") as f:
    norm_val = [line.strip('\n').split() for line in f]

# Split trigrams into individual lists
train_tokens,train_dec_tokens = [[],[],[]],[[],[],[]]
for i in tqdm(range(3), desc="Split train trigrams"):
    train_tokens[i], train_dec_tokens[i] = [tri[i] for tri in train], [tri[i] for tri in norm_train]
val_tokens,val_dec_tokens = [[],[],[]],[[],[],[]]
for i in tqdm(range(3), desc="Split val trigrams"):
    val_tokens[i], val_dec_tokens[i] = [tri[i] for tri in val], [tri[i] for tri in norm_val]

maxlen = max([len(token) for tri in train for token in tri]) + 2

input_chars = set(' '.join(train_tokens[0]+train_tokens[1]+train_tokens[2]) + '*' + '\t')
target_chars = set(' '.join(train_dec_tokens[0]+train_dec_tokens[1]+train_dec_tokens[2]) + '*' + '\t')
nb_input_chars = len(input_chars)
nb_target_chars = len(target_chars)
# Define training and evaluation configuration.
input_ctable  = CharacterTable(input_chars)
target_ctable = CharacterTable(target_chars)

train_steps = len(train_tokens[0]) // train_batch_size
val_steps = len(val_tokens[0]) // val_batch_size
print("Number of train_steps:",train_steps)
print("Number of val_steps:",val_steps)


# Compile the model.
model, encoder_model, decoder_model = seq2seq(hidden_size, nb_input_chars, nb_target_chars)
model_cnt=0
print(model.summary())

# Train and evaluate.
for epoch in range(model_cnt,100):
    print('Main Epoch {:d}/{:d}'.format(epoch + 1, nb_epochs))

    st_ind,et_ind = int(len(train_tokens[0]) * (epoch/100)), int(len(train_tokens[0])*((epoch+1)/100))
    sv_ind,ev_ind = int(len(val_tokens[0]) * (epoch/100)), int(len(val_tokens[0])*((epoch+1)/100))

    train_encoder,train_decoder,train_target = [[],[],[]],[[],[],[]],[[],[],[]]
    for i in range(3):
        train_encoder[i], train_decoder[i], train_target[i] = transform2(tokens=train_tokens[i],maxlen=maxlen, dec_tokens=train_dec_tokens[i], reverse=reverse)

    train_encoder_batch = batch_triplet(train_encoder, maxlen, input_ctable, train_batch_size)
    train_decoder_batch = batch_triplet(train_decoder, maxlen, target_ctable, train_batch_size)
    train_target_batch  = batch_triplet(train_target, maxlen, target_ctable, train_batch_size)

    train_loader = datagen_triplet(train_encoder_batch, train_decoder_batch, train_target_batch)

    val_encoder,val_decoder,val_target = [[],[],[]],[[],[],[]],[[],[],[]]
    for i in range(3):
        val_encoder[i], val_decoder[i], val_target[i] = transform2(tokens=val_tokens[i],maxlen=maxlen, dec_tokens=val_dec_tokens[i], reverse=True)

    val_encoder_batch = batch_triplet(val_encoder, maxlen, input_ctable, val_batch_size)
    val_decoder_batch = batch_triplet(val_decoder, maxlen, target_ctable, val_batch_size)
    val_target_batch  = batch_triplet(val_target, maxlen, target_ctable, val_batch_size)

    val_loader = datagen_triplet(val_encoder_batch, val_decoder_batch, val_target_batch)
    
    history = model.fit(train_loader,
                    steps_per_epoch=train_steps,
                    epochs=1, verbose=1,
                    validation_data=val_loader,
                    validation_steps=val_steps)

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

