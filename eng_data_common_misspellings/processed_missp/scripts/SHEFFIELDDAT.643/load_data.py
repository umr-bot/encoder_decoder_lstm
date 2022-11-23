# coding: utf-8
with open("mispelling_files/SHEFFIELDDAT.643") as f:
    data1 = [line.strip('\n').lower().split() for line in f]
# tok[1] is err and tok[0] is norm
with open("err_norm_data/SHEFFIELDDAT.643",'w') as f:
    for tup in data1: f.write(tup[1]+','+tup[0]+'\n')


