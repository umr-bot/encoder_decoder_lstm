# coding: utf-8
with open("mispelling_files/GATESDAT.643") as f:
    data = [line.strip('\n') for line in f]

err_norms= []
for line in data:
    # Match all digits in the string and replace them with an empty string
    new_string = line.translate({ord(c): '' for c in '*$0123456789'})
    toks = new_string.split()
    if len(toks) < 2: continue
    for cnt in range(1,len(toks)):
        # pair first token with each other token in toks
        err_norms.append((toks[0], toks[cnt]))
# tok[1] is err and tok[0] is norm
# uncomment below if wanting to write data out to file
with open("err_norm_data/GATESDAT.643",'w') as f:
    for tup in err_norms:
        f.write(tup[1]+','+tup[0]+'\n')
        
