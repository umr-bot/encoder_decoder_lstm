import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Layer,dot, Activation, concatenate
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

class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units, verbose=0):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)
    self.verbose= verbose

  def call(self, query, values):
    if self.verbose:
      print('\n******* Bahdanau Attention STARTS******')
      print('query (decoder hidden state): (batch_size, hidden size) ', query.shape)
      print('values (encoder all hidden state): (batch_size, max_len, hidden size) ', values.shape)

    # query hidden state shape == (batch_size, hidden size)
    # query_with_time_axis shape == (batch_size, 1, hidden size)
    # values shape == (batch_size, max_len, hidden size)
    # we are doing this to broadcast addition along the time axis to calculate the score
    query_with_time_axis = tf.expand_dims(query, 1)
    
    if self.verbose:
      print('query_with_time_axis:(batch_size, 1, hidden size) ', query_with_time_axis.shape)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(
        self.W1(query_with_time_axis) + self.W2(values)))
    if self.verbose:
      print('score: (batch_size, max_length, 1) ',score.shape)
    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)
    if self.verbose:
      print('attention_weights: (batch_size, max_length, 1) ',attention_weights.shape)
    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    if self.verbose:
      print('context_vector before reduce_sum: (batch_size, max_length, hidden_size) ',context_vector.shape)
    context_vector = tf.reduce_sum(context_vector, axis=1)
    if self.verbose:
      print('context_vector after reduce_sum: (batch_size, hidden_size) ',context_vector.shape)
      print('\n******* Bahdanau Attention ENDS******')
    return context_vector, attention_weights

@tf.keras.utils.register_keras_serializable()
class AttentionLayer(Layer):

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        #config["mask"] = self.mask
        return config

    def compute_mask(self, inputs, mask=None):
        #self.mask = mask
        if mask == None:
            return None
        return mask[1]
    def compute_output_shape(self, input_shape):
        return (input_shape[1][0],input_shape[1][1],input_shape[1][2]*2)

    def call(self, inputs, mask=None):
        encoder_outputs, decoder_outputs = inputs

        """
        Task 3 attention

        Start
        """
        luong_score = tf.matmul(decoder_outputs, encoder_outputs, transpose_b=True)
        alignment = tf.nn.softmax(luong_score, axis=2)
        context = tf.matmul(K.expand_dims(alignment,axis=2), K.expand_dims(encoder_outputs,axis=1))
        encoder_vector = K.squeeze(context,axis=2)

        """
        End Task 3
        """
        # [batch,max_dec,2*emb]
        new_decoder_outputs = K.concatenate([decoder_outputs, encoder_vector])
        
        return new_decoder_outputs

def luong_attention(encoder_outputs, decoder_outputs):
    print(encoder_outputs.shape)
    print(decoder_outputs.shape)
    luong_score = tf.matmul(decoder_outputs, encoder_outputs, transpose_b=True)

    alignment = Activation('softmax')(luong_score)

    #context = dot([attention, encoder_outputs],axes=[2,1])
    context = tf.matmul(alignment, K.expand_dims(encoder_outputs,axis=0))
    #context = BatchNormalization(momentum=0.6)(context)

    decoder_combined_context = concatenate([context, decoder_outputs])
    return decoder_combined_context

def luong_attention2(encoder_outputs, decoder_outputs):
    luong_score = tf.matmul(decoder_outputs, encoder_outputs, transpose_b=True)
    alignment = tf.nn.softmax(luong_score, axis=2)
    context = tf.matmul(K.expand_dims(alignment,axis=2), K.expand_dims(encoder_outputs,axis=1))
    encoder_vector = K.squeeze(context,axis=2)

    new_decoder_outputs = K.concatenate([decoder_outputs, encoder_vector])
    
    return new_decoder_outputs


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
    #encoder_outputs, state_h, state_c = encoder_lstm_1(encoder_inputs)

    encoder_lstm_2 = LSTM(hidden_size, recurrent_dropout=0.2,
                        return_sequences=False, return_state=True,
                        name='encoder_lstm_2')
    #NOTE: last lstm layer should have return_state=True
    # here encoder outputs contains all state_h 's including the returned one
    encoder_outputs_2, encoder_state_h, encoder_state_c = encoder_lstm_2(encoder_outputs_1)

    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [encoder_state_h, encoder_state_c]
    
    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, nb_target_chars),
                           name='decoder_data')
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the return
    # states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(hidden_size, dropout=0.2, return_sequences=True,
                        return_state=True, name='decoder_lstm')
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

    # Set up the attention layer -------------------------------------------
    #attention= BahdanauAttention(hidden_size) # Bahdanau
    #context_vector, attention_weights=attention(decoder_outputs, encoder_outputs_2)
    #context_vector = tf.expand_dims(context_vector, axis=1)
    #decoder_inputs = tf.expand_dims(decoder_inputs, axis=1)
    #decoder_attention_inputs = tf.concat([context_vector, decoder_inputs], axis=-1)
    #decoder_attention_inputs = tf.expand_dims(decoder_attention_inputs, 1)
    #decoder_outputs, _, _ = decoder_lstm(decoder_attention_inputs)
    # ----------------------------------------------------------------------
    #decoder_attention = AttentionLayer() #Luong
    #decoder_outputs = decoder_attention([encoder_outputs_2,decoder_outputs])
    # --------------------------------------------
#    attention = dot([decoder_outputs, encoder_outputs_2], axes=[2, 2])
#    attention = Activation('softmax')(attention)
#
#    context = dot([attention, encoder_outputs_2], axes=[2,1])
#    context = BatchNormalization(momentum=0.6)(context)
#
#    decoder_combined_context = concatenate([context, decoder_outputs])
    decoder_combined_context = luong_attention(encoder_outputs_2,decoder_outputs)   
    #---------------------------------------------------------------------------    
    decoder_softmax = Dense(nb_target_chars, activation='softmax', name='decoder_softmax')
    decoder_outputs = decoder_softmax(decoder_combined_context)

    # The main model will turn `encoder_input_data` & `decoder_input_data`
    # into `decoder_target_data`
    model = Model(inputs=[encoder_inputs, decoder_inputs],
                  outputs=decoder_outputs)
    
    #adam = tensorflow.keras.optimizers.Adam(lr=0.001, decay=0.0)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', recall, f1_score])
                  #metrics=['accuracy', truncated_acc, truncated_loss, recall, precision, f1_score])
    ################################################################################### 
    # The encoder_model and decoder_models defined below are used when evaluating/using model
    # Define the encoder model separately.
    encoder_model = Model(inputs=encoder_inputs, outputs=[encoder_outputs_2,encoder_states]) # for Luong attention
    #encoder_model = Model(inputs=encoder_inputs, outputs=encoder_states) # defualt enc-dec encoder

#    decoder_state_input_h = Input(shape=(hidden_size,))
#    decoder_state_input_c = Input(shape=(hidden_size,))
#    encoder_outputs_input = Input(shape=(None,hidden_size,))
#
#    """
#    Task 2 decoder for inference
#
#    Start
#    """
#    decoder_states_input = [decoder_state_input_h, decoder_state_input_c]
#    decoder_outputs_test,decoder_state_output_h,decoder_state_output_c = decoder_lstm(decoder_inputs,initial_state=decoder_states_input)
#    decoder_states_output = [decoder_state_output_h, decoder_state_output_c]
#    decoder_attention = AttentionLayer()
#    decoder_outputs_test = decoder_attention([encoder_outputs_input,decoder_outputs_test])
#
#    decoder_outputs_test = decoder_softmax(decoder_outputs_test)
#
#    """
#    End Task 2
#    """
#
#    decoder_model = Model([decoder_inputs,decoder_state_input_h,decoder_state_input_c,encoder_outputs_input],
#                          [decoder_outputs_test,decoder_state_output_h,decoder_state_output_c])
    ##########################################################################
    #decoder_model = Model(inputs=[decoder_inputs] + decoder_states_input,
    #                      outputs=[decoder_outputs_test] + decoder_states_output)

    ####################################################################
    # Define the decoder model separately.
    decoder_state_input_h = Input(shape=(hidden_size,))
    decoder_state_input_c = Input(shape=(hidden_size,))
    encoder_outputs_input = Input(shape=(hidden_size,))
    #decoder_state_input_h, decoder_state_input_c comes from encoder lstm layer 2 output
    decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]
    
    # only initial state uses encoder lstm layer 2 outputs
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_state_inputs)
    decoder_states = [state_h, state_c]
    #------------------------------------------------------------------------------
    decoder_combined_context = luong_attention(encoder_outputs_input,decoder_outputs)
    #-------------------------------------------------------------------------------
    decoder_outputs = decoder_softmax(decoder_combined_context)
    
    # total inputs to decoder = decoder_inputs + encoder_lstm_layer_2_outputs
    #decoder_model = Model(inputs=[decoder_inputs] + decoder_states_inputs,
    #                      outputs=[decoder_outputs] + decoder_states)
    decoder_model = Model([decoder_inputs,decoder_state_inputs,encoder_outputs_input],
                          [decoder_outputs,decoder_states])

    #####################################################################################
    return model, encoder_model, decoder_model
