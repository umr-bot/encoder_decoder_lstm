# coding: utf-8
with open("mispelling_files/FAWTHROP1DAT.643") as f:
    data1 = [line.strip('\n').lower().split() for line in f]
with open("mispelling_files/FAWTHROP2DAT.643") as f:
    data2 = [line.strip('\n').lower().split() for line in f]
# tup[1] is err and tup[0] is norm
with open("err_norm_data/FAWTHROP1DAT.643",'w') as f:
    for tup in data1: f.write(tup[1]+','+tup[0]+'\n')
with open("err_norm_data/FAWTHROP2DAT.643",'w') as f:
    for tup in data2: f.write(tup[1]+','+tup[0]+'\n')


