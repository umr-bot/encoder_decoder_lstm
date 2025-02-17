import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Layer, Lambda
from tensorflow.keras import optimizers, metrics, backend as K
import numpy as np

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

def seq2seq_bahdanau(latentSpaceDimension, n_timesteps_in, n_features, verbose=0):
    #verbose= 0
    #See all debug messages

    batch_size=1
    if verbose:
      print('***** Model Hyper Parameters *******')
      print('latentSpaceDimension: ', latentSpaceDimension)
      print('batch_size: ', batch_size)
      print('sequence length: ', n_timesteps_in)
      print('n_features: ', n_features)

      print('\n***** TENSOR DIMENSIONS *******')

    # The first part is encoder
    #encoder_inputs = Input(shape=(n_timesteps_in, n_features), name='encoder_inputs')
    encoder_inputs = Input(shape=(None, n_features), name='encoder_inputs')

    encoder_lstm = LSTM(latentSpaceDimension,return_sequences=True, return_state=True,  name='encoder_lstm')
    encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_inputs)

    if verbose:
      print ('Encoder output shape: (batch size, sequence length, latentSpaceDimension) {}'.format(encoder_outputs.shape))
      print ('Encoder Hidden state shape: (batch size, latentSpaceDimension) {}'.format(encoder_state_h.shape))
      print ('Encoder Cell state shape: (batch size, latentSpaceDimension) {}'.format(encoder_state_c.shape))
    # initial context vector is the states of the encoder
    encoder_states = [encoder_state_h, encoder_state_c]
    if verbose:
      print(encoder_states)
    # Set up the attention layer
    attention= BahdanauAttention(latentSpaceDimension, verbose=verbose)


    # Set up the decoder layers
    #decoder_inputs = Input(shape=(1, (n_features+latentSpaceDimension)),name='decoder_inputs')
    decoder_inputs = Input(shape=(None,n_features),name="decoder_inputs")
    decoder_lstm = LSTM(latentSpaceDimension,  return_state=True, name='decoder_lstm')
    decoder_dense = Dense(n_features, activation='softmax',  name='decoder_dense')

    all_outputs = []

    # 1 initial decoder's input data
    # Prepare initial decoder input data that just contains the start character
    # Note that we made it a constant one-hot-encoded in the model
    # that is, [1 0 0 0 0 0 0 0 0 0] is the first input for each loop
    # one-hot encoded zero(0) is the start symbol
    #inputs = np.zeros((batch_size, 1, n_features))
    #inputs = tf.expand_dims(encoder_inputs,axis=1)
    inputs = encoder_inputs
    #inputs[:, 0, 0] = 1


    # 2 initial decoder's state
    # encoder's last hidden state + last cell state
    decoder_outputs = encoder_state_h
    states = encoder_states
    if verbose:
      print('initial decoder inputs: ', inputs.shape)

    # decoder will only process one time step at a time.
    for _ in range(n_timesteps_in):

        # 3 pay attention
        # create the context vector by applying attention to
        # decoder_outputs (last hidden state) + encoder_outputs (all hidden states)
        context_vector, attention_weights=attention(decoder_outputs, encoder_outputs)
        if verbose:
          print("Attention context_vector: (batch size, units) {}".format(context_vector.shape))
          print("Attention weights : (batch_size, sequence_length, 1) {}".format(attention_weights.shape))
          print('decoder_outputs: (batch_size,  latentSpaceDimension) ', decoder_outputs.shape )

        context_vector = tf.expand_dims(context_vector, axis=1)
        if verbose:
          print('Reshaped context_vector: ', context_vector.shape )
        context_vector = tf.cast(context_vector, dtype = tf.float32) # fixes some bug in keras for float casting
        inputs = tf.cast(inputs, dtype = tf.float32)
        # 4. concatenate the input + context vectore to find the next decoder's input
        inputs = tf.concat([context_vector, inputs], axis=-1)

        if verbose:
          print('After concat inputs: (batch_size, 1, n_features + hidden_size): ',inputs.shape )

        # 5. passing the concatenated vector to the LSTM
        # Run the decoder on one timestep with attended input and previous states
        decoder_outputs, state_h, state_c = decoder_lstm(inputs, initial_state=states)
        #decoder_outputs = tf.reshape(decoder_outputs, (-1, decoder_outputs.shape[2]))

        outputs = decoder_dense(decoder_outputs)
        # 6. Use the last hidden state for prediction the output
        # save the current prediction
        # we will concatenate all predictions later
        outputs = tf.expand_dims(outputs, 1)
        all_outputs.append(outputs)
        # 7. Reinject the output (prediction) as inputs for the next loop iteration
        # as well as update the states
        inputs = outputs
        states = [state_h, state_c]


    # 8. After running Decoder for max time steps
    # we had created a predition list for the output sequence
    # convert the list to output array by Concatenating all predictions
    # such as [batch_size, timesteps, features]
    decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)

    # 9. Define and compile model
    #model_encoder_decoder_Bahdanau_Attention = Model(encoder_inputs, decoder_outputs, name='model_encoder_decoder')
    model_encoder_decoder_Bahdanau_Attention = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs, name="model_encoder_decoder")

    model_encoder_decoder_Bahdanau_Attention.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    # Define the encoder model separately
    #encoder_model = Model(inputs=encoder_inputs, outputs=encoder_states)
    return model_encoder_decoder_Bahdanau_Attention
