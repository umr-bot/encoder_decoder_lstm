# coding: utf-8
with open("eng_za") as f:
    text=[]
    for line in f:
        text.append([tok for tok in line.strip('\n').split() if set(",()[]-<>_;.?!1234567890").isdisjoint(set(tok))])
with open("eng_za",'w') as f:
    for line in text: f.write(" ".join(tok for tok in line) + '\n')

tris = []
for line in text:
    for i in range(len(line)-2):
        tris.append((line[i],line[i+1],line[i+2]))

#tris = list(set(tris))
#
#num_folds = 5
#parts = [[] for i in range(num_folds)]
#div = len(tris)//num_folds
#
#parts[0] = tris[0:div]
#parts[1] = tris[div:div*2]
#parts[2] = tris[div*2:div*3]
#parts[3] = tris[div*3:div*4]
#parts[4] = tris[div*4:]
#for i in range(num_folds):
#    with open("folds/part"+str(i+1),'w') as f:
#        for tri in parts[i]: f.write(tri[0]+','+tri[1]+','+tri[2] + '\n')
#
