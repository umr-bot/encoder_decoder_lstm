# coding: utf-8
from collections import Counter, defaultdict
from tqdm import tqdm

def count_tup_errs(tups):
    err_cnt = 0
    for tok, norm_tok in tups:
        if tok!=norm_tok: err_cnt+=1
    return err_cnt
train,val,test=[],[],[]
for i in range(4):
    with open("folds/foldset"+str(i+1) +"/train") as f:
        train.append([line.strip('\n') for line in f])
    with open("folds/foldset"+str(i+1) +"/val") as f:
        val.append([line.strip('\n') for line in f])
    with open("folds/foldset"+str(i+1) +"/test") as f:
        test.append([line.strip('\n') for line in f])
norm_train,norm_val,norm_test=[],[],[]
for i in range(4):
    with open("folds/norm_foldset"+str(i+1) +"/train") as f:
        norm_train.append([line.strip('\n') for line in f])
    with open("folds/norm_foldset"+str(i+1) +"/val") as f:
        norm_val.append([line.strip('\n') for line in f])
    with open("folds/norm_foldset"+str(i+1) +"/test") as f:
        norm_test.append([line.strip('\n') for line in f])
# Confirm that no errors 
#for i in range(4):
#    print(f"Train[{i}] errors {count_tup_errs(zip(train[i],norm_train[i]))}")
#    print(f"Val[{i}] errors {count_tup_errs(zip(val[i],norm_val[i]))}")
#    print(f"Test[{i}] errors {count_tup_errs(zip(test[i],norm_test[i]))}")

with open("wiki_missp_aspell.txt") as f:
    missp = [(tups.split(',')[0].strip(' '),tups.split(',')[1].strip(' \n')) for tups in f]
errors = list(zip(*missp))[0]
norms = list(zip(*missp))[1] # get all normalized tokens of missplelled words
norm_err_dict = dict(zip(norms,errors)) # create zip of reversed order of errors and norms
d = defaultdict(list)
for tok,norm_tok in zip(errors,norms):
    d[norm_tok].append(tok)

counts = Counter(norms) # bin normalized misspelled word tokens

interval = len(missp)/(8*3) # 8*3 is from train, val and norm partitions
cnt = 0
nt=[defaultdict(list) for dummy in range(4)]

for d_norm_tok,err_list in tqdm(d.items(), desc = "Looping over d"):
    for i in range(4):
        for norm_tok in set(norm_train[i]):
            if norm_tok == d_norm_tok:
                nt[i][norm_tok].append(err_list)
                cnt+=len(err_list)
        if cnt > interval:
            cnt=0
            continue

