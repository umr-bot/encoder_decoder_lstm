# coding: utf-8
from utils import CharacterTable
import numpy as np
batch_size=10
with open("eng_data_common_misspellings/trigrams/fold1/val") as f:
    val = [line.strip('\n').split() for line in f]
maxlen = max([len(tok) for tri in val for tok in tri])
tokens = [tok for tri in val for tok in tri]
chars = set(' '.join(tokens))
ctable  = CharacterTable(chars)
def batch_triplet(token_triplets, maxlen, ctable, batch_size=128):
    def generate(token_triplets):
        while(True): # This flag yields an infinite generator.
            for token_triplet in token_triplets: yield token_triplet
    token_iterator = generate(val)
    data_batch = np.zeros((batch_size, 3, maxlen, ctable.size), dtype=np.float32)
    while(True):
        for i in range(batch_size):
            token = next(token_iterator)
            data_batch[i] = (ctable.encode(token[0], maxlen), ctable.encode(token[1], maxlen),ctable.encode(token[2], maxlen))
            #except: print(token)
        yield data_batch
x=batch_triplet(token_triplets=val,maxlen=maxlen,ctable=ctable,batch_size=batch_size)
