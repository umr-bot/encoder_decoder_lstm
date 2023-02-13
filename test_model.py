from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate, concatenate
from model import recall, f1_score

#hidden_size, nb_input_chars, nb_target_chars=100,17,10
def seq2seq(hidden_size, nb_input_chars, nb_target_chars):
# Define the main model consisting of encoder and decoder.
    encoder_inputs_1 = Input(shape=(None, nb_input_chars), name='encoder_data_1')
    encoder_inputs_2 = Input(shape=(None, nb_input_chars), name='encoder_data_2')
    encoder_input_concat = concatenate([encoder_inputs_1,encoder_inputs_2]) # concat along rows

    encoder_lstm_1 = LSTM(hidden_size, recurrent_dropout=0.2, return_sequences=True, return_state=True, name='encoder_lstm_1')
# here encoder outputs contains three things:{(all h states),(last h state),(last c state)}
#encoder_outputs_1 = encoder_lstm_1(encoder_inputs_1)
    encoder_h_outputs_1, state_h_not_used, state_c_not_used = encoder_lstm_1(encoder_input_concat)

    encoder_lstm_2 = LSTM(hidden_size, recurrent_dropout=0.2,
                        return_sequences=False, return_state=True,
                        name='encoder_lstm_2')
#NOTE: last lstm layer should have return_state=True
# here encoder outputs contains all state_h 's including the returned one
    encoder_outputs_2, state_h, state_c = encoder_lstm_2(encoder_h_outputs_1)

# We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

# The encoder_model and decoder_models defined below are used when evaluating/using model
# Define the encoder model separately.
    encoder_model = Model(inputs=[encoder_inputs_1, encoder_inputs_2], outputs=encoder_states, name="encoder_outputs_2_model")
#encoder_model.summary()

# Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, nb_target_chars), name='decoder_data')
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the return
# states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(hidden_size, dropout=0.2, return_sequences=True, return_state=True, name='decoder_lstm')
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_softmax = Dense(nb_target_chars, activation='softmax', name='decoder_softmax')
    decoder_outputs = decoder_softmax(decoder_outputs)

# The main model will turn `encoder_input_data` & `decoder_input_data`
# into `decoder_target_data`
    model = Model(inputs=[encoder_inputs_1,encoder_inputs_2, decoder_inputs], outputs=decoder_outputs, name="main_model")

#adam = tensorflow.keras.optimizers.Adam(lr=0.001, decay=0.0)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', recall, f1_score])
                  #metrics=['accuracy', truncated_acc, truncated_loss, recall, precision, f1_score])
####################################################################
# Define the decoder model separately.
    decoder_state_input_h = Input(shape=(hidden_size,))
    decoder_state_input_c = Input(shape=(hidden_size,))
#decoder_state_input_h, decoder_state_input_c comes from encoder lstm layer 2 output
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# only initial state uses encoder lstm layer 2 outputs
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_softmax(decoder_outputs)

# total inputs to decoder = decoder_inputs + encoder_lstm_layer_2_outputs
    decoder_model = Model(inputs=[decoder_inputs] + decoder_states_inputs,
                          outputs=[decoder_outputs] + decoder_states, name="decoder_model")

    return model, encoder_model, decoder_model
