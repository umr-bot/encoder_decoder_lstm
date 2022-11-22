# coding: utf-8
with open("mispelling_files/APPLING2DAT.643") as f:
    data = [line for line in f]
    
X=[]
for line in data:
    if "$" in line: continue
    toks = line.split()
    X.append((toks[0],toks[1]))
    
with open("err_norm_data/APPLING2DAT_643",'w') as f:
    for tup in X: f.write(tup[0]+','+tup[1]+'\n')
    
