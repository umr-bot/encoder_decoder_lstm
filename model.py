import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras import optimizers, metrics, backend as K
# For use with truncated metrics,
# take maxlen from the validation set.
# Hacky and hard-coded for now.
VAL_MAXLEN = 16

def truncated_acc(y_true, y_pred):
#    print("TA y_true len",len(y_true))
#    print("TA y_true\n",y_true)
    y_true = y_true[:, :VAL_MAXLEN, :]
    y_pred = y_pred[:, :VAL_MAXLEN, :]
    
    acc = metrics.categorical_accuracy(y_true, y_pred)
    return K.mean(acc, axis=-1)


def truncated_loss(y_true, y_pred):
#    print("TL y_true len",len(y_true))
#    print("TL y_true\n",y_true)
    y_true = y_true[:, :VAL_MAXLEN, :]
    y_pred = y_pred[:, :VAL_MAXLEN, :]
    
    loss = K.categorical_crossentropy(
        target=y_true, output=y_pred, from_logits=False)
    return K.mean(loss, axis=-1)

def recall(y_true, y_pred):
    y_true = K.ones_like(y_true)
    # K.clip is used in the case of non-binary classification
    # where elements of y_true and y_pred some floating point number
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall = true_positives / (all_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    y_true = K.ones_like(y_true)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2*((prec*rec)/(prec+rec+K.epsilon()))

#class MulticlassTruePositives(tensorflow.keras.metrics.Metric):
#    def __init__(self, name='multiclass_true_positives', **kwargs):
#        super(MulticlassTruePositives, self).__init__(name=name, **kwargs)
#        self.true_positives = self.add_weight(name='tp', initializer='zeros')
#
#    def update_state(self, y_true, y_pred, sample_weight=None):
#        y_pred = tensorflow.reshape(tensorflow.argmax(y_pred, axis=1), shape=(-1, 1, 1))
#        values = tensorflow.cast(y_true, 'int32') == tensorflow.cast(y_pred, 'int32')
#        values = tensorflow.cast(values, 'float32')
#        if sample_weight is not None:
#            sample_weight = tensorflow.cast(sample_weight, 'float32')
#            values = tensorflow.multiply(values, sample_weight)
#        self.true_positives.assign_add(tensorflow.reduce_sum(values))
#
#    def result(self):
#        return self.true_positives
#
#    def reset_states(self):
#        # The state of the metric will be reset at the start of each epoch.
#        self.true_positives.assign(0.)

def seq2seq(hidden_size, nb_input_chars, nb_target_chars):
    """Adapted from:
    https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py
    hidden_size: Number of memory cells in encoder and decoder models
    nb_input_chars: number of input sequences, in train_val.py 's case, chars
    nb_target_chars number of output sequences, in train_val.py, it is chars
    """
    
    # Define the main model consisting of encoder and decoder.
    encoder_inputs = Input(shape=(None, nb_input_chars),
                           name='encoder_data')
    encoder_lstm_1 = LSTM(hidden_size, recurrent_dropout=0.2,
                        return_sequences=True, return_state=True,
                        name='encoder_lstm_1')
    # here encoder outputs contains three things:{(all h states),(last h state),(last c state)}
    encoder_outputs_1 = encoder_lstm_1(encoder_inputs)
    encoder_outputs, state_h, state_c = encoder_lstm_1(encoder_inputs)

    encoder_lstm_2 = LSTM(hidden_size, recurrent_dropout=0.2,
                        return_sequences=False, return_state=True,
                        name='encoder_lstm_2')
    #NOTE: last lstm layer should have return_state=True
    # here encoder outputs contains all state_h 's including the returned one
    encoder_outputs_2, state_h, state_c = encoder_lstm_2(encoder_outputs_1)

    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, nb_target_chars),
                           name='decoder_data')
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the return
    # states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(hidden_size, dropout=0.2, return_sequences=True,
                        return_state=True, name='decoder_lstm')
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=encoder_states)
    decoder_softmax = Dense(nb_target_chars, activation='softmax',
                            name='decoder_softmax')
    decoder_outputs = decoder_softmax(decoder_outputs)

    # The main model will turn `encoder_input_data` & `decoder_input_data`
    # into `decoder_target_data`
    model = Model(inputs=[encoder_inputs, decoder_inputs],
                  outputs=decoder_outputs)
    
    #adam = tensorflow.keras.optimizers.Adam(lr=0.001, decay=0.0)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                metrics=['accuracy', recall, f1_score], run_eagerly=True)
                  #metrics=['accuracy', truncated_acc, truncated_loss, recall, precision, f1_score])
    ################################################################################### 
    # The encoder_model and decoder_models defined below are used when evaluating/using model
    # Define the encoder model separately.
    encoder_model = Model(inputs=encoder_inputs, outputs=encoder_states)
    ####################################################################
    # Define the decoder model separately.
    decoder_state_input_h = Input(shape=(hidden_size,))
    decoder_state_input_c = Input(shape=(hidden_size,))
    #decoder_state_input_h, decoder_state_input_c comes from encoder lstm layer 2 output
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    
    # only initial state uses encoder lstm layer 2 outputs
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_softmax(decoder_outputs)
    
    # total inputs to decoder = decoder_inputs + encoder_lstm_layer_2_outputs
    decoder_model = Model(inputs=[decoder_inputs] + decoder_states_inputs,
                          outputs=[decoder_outputs] + decoder_states)
    #####################################################################################
    return model, encoder_model, decoder_model
