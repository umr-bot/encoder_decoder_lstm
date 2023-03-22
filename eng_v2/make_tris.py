# coding: utf-8
from tqdm import tqdm

# Read in text data into list of sentences
with open("eng_za/trigram_tokens") as f:
    tris=[]
    for line in f:
        tris.append([tok for tok in line.strip('\n').split(',') if set("[]-<>_").isdisjoint(set(tok))])
tris = list(set(tuple(tri) for tri in tris)) # only keep types
####################################################
# Create and fill trigram list of all trigrams in text
#tris = []
#for line in text:
#    for i in range(len(line)-2):
#        tris.append((line[i],line[i+1],line[i+2]))
####################################################
# Read in unigram fold data into parts variable
parts=[[],[],[],[],[]]
for i in range(5):
    with open("folds/part"+str(i+1)) as f:
        parts[i] = [tuple(line.strip('\n').split(','))[0] for line in f]
        parts[i] = list(set(parts[i]))
####################################################
# Create folds out of trigrams list and unigram folds stored in parts variable
folds=[[],[],[],[],[]]
for i in range(5):
    for tok in tqdm(parts[i],desc=f"Fold {i}"):
        for tri in tris:
            if tri[1] == tok: folds[i].append(tri)

for i in range(5):
    with open("trigrams/fold"+str(i+1),'w') as f:
        for tri in folds[i]: f.write(tri[0]+','+tri[1]+','+tri[2]+'\n')
# remove trigram  overlap in folds
m = [[] for blank in range(5)]
x = m.copy()
for fold_cnt in range(5): m[fold_cnt] = set(tuple(tri) for tri in folds[fold_cnt])
for i in range(5):
    for j in range(5):
        if m[i].isdisjoint(m[j]) == False and i!=j:
            print("False",end=" ")
            x[i] = m[i] - m[i].intersection(m[j])
    print("------------")
# sanity checkof fold independency
for j in range(5):
    for i in range(5): print(x[j].isdisjoint(x[i]),end=" ")
    print("------------")    

