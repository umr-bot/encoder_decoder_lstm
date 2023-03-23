# coding: utf-8
from tqdm import tqdm
from collections import  Counter

# Read in text data into list of sentences
with open("eng_za/eng_za") as f:
    text=[]
    for line in f:
        text.append([tok for tok in line.strip('\n').split() if set("[]-<>_").isdisjoint(set(tok))])
####################################################
# Create and fill trigram list of all trigrams in text
tris = []
for line in text:
    for i in range(len(line)-2):
        tris.append((line[i],line[i+1],line[i+2]))
####################################################
type_folds=[[] for i in range(5)]
tris_set = list(set(tris))
div = len(tris_set)//5
for i in range(5): type_folds[i] = tris_set[i*div:(i+1)*div]
print("Number of trigram types per fold:",str(div))

cnts = Counter(tris)
folds = [[] for i in range(5)]
for i in range(5):
    for tri in type_folds[i]:
        for blank in range(cnts[tuple(tri)]):
            folds[i].append(tri)
print("Printing number of trigram tokens per fold")
for i in range(5): print(f"Fold {i+1}: {len(folds[i])}") 

# Read in unigram fold data into parts variable
#parts=[[],[],[],[],[]]
#for i in range(5):
#    with open("folds/part"+str(i+1)) as f:
#        parts[i] = [tuple(line.strip('\n').split(','))[1] for line in f]
#        parts[i] = list(set(parts[i]))
#####################################################
## Create folds out of trigrams list and unigram folds stored in parts variable
#folds=[[],[],[],[],[]]
#for i in range(5):
#    for tok in tqdm(parts[i],desc=f"Fold {i}"):
#        for tri in tris:
#            if tri[1] == tok: folds[i].append(tri)
#
for i in range(5):
    with open("trigrams/fold"+str(i+1),'w') as f:
        for tri in folds[i]: f.write(tri[0]+' '+tri[1]+' '+tri[2]+'\n')

