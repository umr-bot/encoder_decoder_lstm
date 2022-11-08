# coding: utf-8
from utils import restore_model

# If you want to save the model graph as an image uncomment the 2 lines below
#from tensorflow import keras
#keras.utils.plot_model(model, "model.png",show_shapes=True)

class GateHandler:
    """ Takes in a tensorflow keras model. Has funcionality to access
        gate weights and biases."""
    def __init__(self):
        pass

    def get_gate_weights(self, model, layer_name="encoder_lstm_1"):
        """ Get gate weights for a lstm layer, with name layer_name."""
        units = int(int(model.get_layer(layer_name).trainable_weights[0].shape[1])/4)
        W = model.get_layer(layer_name).get_weights()[0]
        U = model.get_layer(layer_name).get_weights()[1]
        b = model.get_layer(layer_name).get_weights()[2]

        W_i = W[:, :units]
        W_f = W[:, units: units * 2]
        W_c = W[:, units * 2: units * 3]
        W_o = W[:, units * 3:]
        W_list = [W_i,W_f,W_c,W_o]
        
        U_i = U[:, :units]
        U_f = U[:, units: units * 2]
        U_c = U[:, units * 2: units * 3]
        U_o = U[:, units * 3:]
        U_list = [U_i,U_f,U_c,U_o]
        
        b_i = b[:units]
        b_f = b[units: units * 2]
        b_c = b[units * 2: units * 3]
        b_o = b[units * 3:]
        b_list = [b_i,b_f,b_c,b_o]        
        
        return W_list, U_list, b_list

#if __name__ == "__main__":
#    model_path=""
#    gate_handler = GateHandler(model_path)

# Get single weights(input->lstm) vs time step arrays, is it W_i.T ? Yes it is.
#arr=[]
#for j in W_i.shape[-1]: arr.append(W_i[:,j])
# OR more concisely: arr = W_i.transpose()

