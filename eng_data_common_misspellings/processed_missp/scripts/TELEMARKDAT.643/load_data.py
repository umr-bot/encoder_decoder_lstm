# coding: utf-8
with open("mispelling_files/TELEMARKDAT.643") as f:
    data = [line for line in f]
    
X=[]
for line in data:
    toks = line.split()
    X.append((toks[0],toks[1]))
    
with open("err_norm_data/TELEMARKDAT.643",'w') as f:
    for tup in X: f.write(tup[0]+','+tup[1]+'\n')
    
