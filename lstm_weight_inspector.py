# coding: utf-8
import logging
logging.getLogger('tensorflow').disabled = True
from tqdm import tqdm
from utils import restore_model
from lstm_gate_weights import GateHandler

class lstm_weight_inspector:
    
    def __init__(self):
        pass

    def get_models(self, hidden_size=512, start_model=1, num_models=100):
        """ Get models across epochs."""
        models=[]#,enc_models,dec_models=[],[],[]
        for model_cnt in tqdm(range(start_model,num_models+1), desc="Looping over models...", initial=start_model, total=num_models, unit="models", colour='#00ff00'):
            model, enc_model, dec_model = restore_model("checkpoints/seq2seq_epoch_"+str(model_cnt)+".h5",hidden_size=hidden_size)
            models.append(model)
            #enc_models.append(enc_model)
            #dec_models.append(dec_model)
        return models

if __name__ == "__main__":
    lwi = lstm_weight_inspector()
    models = lwi.get_models(start_model=90, num_models=100)
    gh = GateHandler()
    W,U,b = gh.get_gate_weights(models[0])

