# coding: utf-8
from utils import CharacterTable
from model import recall,precision,f1_score

# Load evaluated/decoded data
with open("toks") as f:
    ll = [line.split(',') for line in f]
in_toks,dec_toks,tar_toks=[],[],[]
for line in ll:
    in_toks.append(line[0])
    dec_toks.append(line[1])
    tar_toks.append(line[2].strip('\n'))

# Define character tables
in_chars = set(' '.join(in_toks) + '*' + '\t')
dec_chars = set(' '.join(dec_toks) + '*' + '\t')
tar_chars = set(' '.join(tar_toks) + '*' + '\t')

in_ctable = CharacterTable(in_chars)
dec_ctable = CharacterTable(dec_chars)
tar_ctable = CharacterTable(tar_chars)
# Max token length used to determine padding for one-hot word encodings
maxlen = max([len(token) for token in in_toks]) + 2
maxlen = max(maxlen, max([len(token) for token in dec_toks]) + 2)
maxlen = max(maxlen, max([len(token) for token in tar_toks]) + 2)

in_toks_enc,dec_toks_enc,tar_toks_enc=[],[],[]
for in_tok,dec_tok,tar_tok in zip(in_toks,dec_toks,tar_toks):
    #in_toks_enc.append(in_ctable.encode(C=in_tok,nb_rows=maxlen))
    dec_toks_enc.append(tar_ctable.encode(C=dec_tok,nb_rows=maxlen))
    tar_toks_enc.append(tar_ctable.encode(C=tar_tok,nb_rows=maxlen))

#prec = precision(dec_toks_enc,tar_toks_enc)
#f1_score = f1_score(dec_toks_enc,tar_toks_enc)

