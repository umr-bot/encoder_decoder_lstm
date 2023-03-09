from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Layer
from tensorflow.keras import optimizers, metrics, backend as K
from model import recall, f1_score

# Add attention layer to the deep learning network
class attention(Layer):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)

    def build(self,input_shape):
        self.W=self.add_weight(name='attention_weight', shape=(input_shape[-1],1), 
                               initializer='random_normal', trainable=True)
        self.b=self.add_weight(name='attention_bias', shape=(input_shape[1],1), 
                               initializer='zeros', trainable=True)        
        super(attention, self).build(input_shape)

    def call(self,x):
        # Alignment scores. Pass them through tanh function
        e = K.tanh(K.dot(x,self.W)+self.b)
        # Remove dimension of size 1
        e = K.squeeze(e, axis=-1)   
        # Compute the weights
        alpha = K.softmax(e)
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = x * alpha
        context = K.sum(context, axis=1)
        return context

def simple_lstm(hidden_size, nb_target_chars):

    decoder_inputs = Input(shape=(None, nb_target_chars),
                           name='decoder_data')
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the return
# states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(hidden_size, return_sequences=True,
                        return_state=True, name='decoder_lstm_1')
    decoder_outputs_1 = decoder_lstm(decoder_inputs)
    # Note stacked LTSM must have last layer with return_Sequences=False
    decoder_lstm = LSTM(hidden_size, return_sequences=True,
                        return_state=False, name='decoder_lstm_2')
    decoder_outputs_2  = decoder_lstm(decoder_outputs_1)
    #decoder_lstm = LSTM(hidden_size, return_sequences=True,
    #                    return_state=False, name='decoder_lstm_3')
    #decoder_outputs_3  = decoder_lstm(decoder_outputs_2)
    #decoder_outputs = attention()(decoder_outputs_3)
    decoder_softmax = Dense(nb_target_chars, activation='softmax',
                            name='decoder_softmax')
    decoder_outputs = decoder_softmax(decoder_outputs_2)

# The main model will turn `encoder_input_data` & `decoder_input_data`
# into `decoder_target_data`
    model = Model(inputs=decoder_inputs,
                  outputs=decoder_outputs)

#adam = tensorflow.keras.optimizers.Adam(lr=0.001, decay=0.0)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                metrics=['accuracy', recall, f1_score])
    return model
