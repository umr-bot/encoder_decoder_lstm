# coding: utf-8
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input
from model_attention import AttentionLayer
from tensorflow.keras import optimizers, metrics, backend as K
from model import f1_score, recall
from utils import CharacterTable, transform2, batch, datagen
import numpy as np
import re
SOS,EOS='\t','*'
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

fold_num=1
# extract training tokens
# train_dec_tokens are targets and train_tokens inputs
with open("eng_v2/folds/fold"+str(fold_num)+"/train") as f:
    train_tups = [line.strip('\n').split(',') for line in f]
train_dec_tokens, train_tokens = zip(*train_tups)

with open("eng_v2/folds/fold"+str(fold_num)+"/val") as f:
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

maxlen = max([len(token) for token in train_tokens]) + 2

# Train and evaluate.
for epoch in range(0,1):
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

    train_loader = datagen(train_encoder_batch,train_decoder_batch,train_target_batch)
    val_loader = datagen(val_encoder_batch, val_decoder_batch, val_target_batch)
    

# Compile the model.
custom_objects = {"Attention_layer":AttentionLayer, "recall": recall, "f1_score":f1_score}
model = load_model("eng_luong_checkpoints_fold1/seq2seq_epoch_1.h5",custom_objects=custom_objects)
print("Printing out main model layer details: layer_name,layer_shape,layer_is_trainable")
for i, layer in enumerate(model.layers):
    print(i, layer.name, layer.output_shape, layer.trainable)

lstm_2_flag=True
encoder_inputs = model.input[0] # encoder_data
encoder_lstm1 = model.get_layer('encoder_lstm_1')
if lstm_2_flag==True: encoder_lstm2 = model.get_layer('encoder_lstm_2')

encoder_outputs_1 = encoder_lstm1(encoder_inputs)
if lstm_2_flag==True: encoder_outputs, state_h, state_c = encoder_lstm2(encoder_outputs_1)
else: encoder_outputs, state_h, state_c = encoder_outputs_1

encoder_output_states = [state_h, state_c]
encoder_model = Model(inputs=encoder_inputs, outputs=[encoder_outputs,encoder_output_states])

decoder_inputs = model.input[1] # decoder_data
decoder_state_input_h = Input(shape=(hidden_size,))
decoder_state_input_c = Input(shape=(hidden_size,))
# only used to connect encoder_states into decoder model
encoder_output_inputs = Input(shape=(hidden_size,))
decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_lstm = model.get_layer('decoder_lstm')
decoder_outputs, decoder_state_output_h, decoder_state_output_c = decoder_lstm(decoder_inputs, initial_state=decoder_state_inputs)
#decoder_states = [decoder_state_input_h, decoder_state_input_c]
#if 1==1:
#    #decoder_attention = AttentionLayer() #Luong
#    decoder_attention = model.get_layer("attention_layer")
#    decoder_outputs = decoder_attention([encoder_outputs[0],decoder_outputs])
attention_matmul = model.get_layer("tf.linalg.matmul")
luong_score = attention_matmul(decoder_outputs, encoder_output_inputs, transpose_b=True)
attention_softmax = model.get_layer("activation")
alignment = attention_softmax(luong_score)
attention_matmul_1 = model.get_layer("tf.linalg.matmul_1")
#attention_expand_dims = model.get_layer("tf.exapand_dims")
context = attention_matmul_1(alignment, K.expand_dims(encoder_output_inputs,axis=0))
attention_concatenate = model.get_layer("concatenate")
decoder_combined_context = attention_concatenate([context, decoder_outputs])

decoder_softmax = model.get_layer('decoder_softmax')
decoder_outputs = decoder_softmax(decoder_combined_context)

decoder_model = Model([decoder_inputs,decoder_state_inputs,encoder_output_inputs], [decoder_outputs,decoder_state_output_h, decoder_state_output_c])

input_tokens, target_tokens = [],[]
nb_examples, inputs, targets = 5, val_encoder, val_target
for index in range(nb_examples):
    input_tokens.append(inputs[index])
    target_tokens.append(targets[index])

input_sequences = batch(input_tokens, maxlen, input_ctable, nb_examples, reverse)
input_sequences = next(input_sequences)
# Encode the input as state vectors.
encoder_output_inference,states_value = encoder_model.predict(input_sequences)
# Create batch of empty target sequences of length 1 character.
target_sequences = np.zeros((nb_examples, 1, target_ctable.size))
target_sequences[:, 0, target_ctable.char2index['\t']] = 1.0
#context_vector, attention_weights=attention(states_value[1],target_sequences)#att(enc_out,dec_out)

# Sampling loop for a batch of sequences.
# Exit condition: either hit max character limit
# or encounter end-of-sequence character.
decoded_tokens = [''] * nb_examples
for _ in range(maxlen):
    # `char_probs` has shape
    # (nb_examples, 1, nb_target_chars)
    char_probs, h, c = decoder_model.predict([target_sequences,states_value,encoder_output_inference])

    # Reset the target sequences.
    target_sequences = np.zeros((nb_examples, 1, target_ctable.size))

    # Sample next character using argmax or multinomial mode.
    sampled_chars = []
    for i in range(nb_examples):
        if sample_mode == 'argmax':
            next_index, next_char = target_ctable.decode(
                    char_probs[i], calc_argmax=True)
        elif sample_mode == 'multinomial':
            next_index, next_char = target_ctable.sample_multinomial(
                    char_probs[i], temperature=0.5)
        else:
            raise Exception(
                    "`sample_mode` accepts `argmax` or `multinomial`.")
        decoded_tokens[i] += next_char
        sampled_chars.append(next_char) 
        # Update target sequence with index of next character.
        target_sequences[i, 0, next_index] = 1.0

    stop_char = set(sampled_chars)
    if len(stop_char) == 1 and stop_char.pop() == EOS: break

    # Update states.
    states_value = [h, c]

# Sampling finished.
input_tokens   = [re.sub('[%s]' % EOS, '', token)
        for token in input_tokens]
target_tokens  = [re.sub('[%s]' % EOS, '', token)
        for token in target_tokens]
decoded_tokens = [re.sub('[%s]' % EOS, '', token)
        for token in decoded_tokens]
print(f"input tokens:\n {input_tokens}")
print(f"decoded tokens:\n {decoded_tokens}")
print(f"target tokens:\n {target_tokens}")
