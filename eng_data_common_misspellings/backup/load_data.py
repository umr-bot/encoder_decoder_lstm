# coding: utf-8
from collections import defaultdict
from tqdm import tqdm
train,val,test=[[] for dummy in range(4)],[[] for dummy in range(4)],[[] for dummy in range(4)]
train_set,val_set,test_set=[set() for dummy in range(4)],[set() for dummy in range(4)],[set() for dummy in range(4)]
for i in range(4):
    with open("disjoint_folds/train_"+str(i+1)) as f:
        for line in tqdm(f, desc=f"train {i+1}"):
            train[i].append(line.strip('\n'))
            train_set[i].add(line.split(',')[0])
    with open("disjoint_folds/val_"+str(i+1)) as f:
        for line in tqdm(f, desc=f"val {i+1}"):
            val[i].append(line.strip('\n'))
            val_set[i].add(line.split(',')[0])
    with open("disjoint_folds/test_"+str(i+1)) as f:
        for line in tqdm(f, desc=f"test {i+1}"):
            test[i].append(line.strip('\n'))
            test_set[i].add(line.split(',')[0])

def make_disjoint(lst, set_, data_desc):
    """ Make a 4-fold list containing 2-tuples on each line in each fold,
        have disjoint folds by using the sets of each fold. The first element
        of the tuple is what is being used to make the folds independent.
        lst: a list containing the folds with each line containing a 2-tuple
        set_: the set of tokens (types) in a folds that taken across the first 
              element in the 2-tuple in each fold."""
    dis = [[] for dummy in range(4)]
    dis[0] = [toks for toks in tqdm(lst[0],desc=f"{data_desc} fold 0") if toks.split(',')[0] not in set_[1].union(set_[2],set_[3])]
    dis[1] = [toks for toks in tqdm(lst[1],desc=f"{data_desc} fold 1") if toks.split(',')[0] not in train_set[0].union(set_[2],set_[3])]
    dis[2] = [toks for toks in tqdm(lst[2],desc=f"{data_desc} fold 2") if toks.split(',')[0] not in train_set[0].union(set_[1],set_[3])]
    dis[3] = [toks for toks in tqdm(lst[3],desc=f"{data_desc} fold 3") if toks.split(',')[0] not in train_set[0].union(set_[1],set_[2])]
    return dis
# Make train val and test disjoint if not already
#disjoint_train = make_disjoint(train, train_set, data_desc="train")
#disjoint_val = make_disjoint(val, val_set, data_desc="val")
#disjoint_test = make_disjoint(test, test_set, data_desc="test")

print("Tokens")
for i in range(4):
    print(len(train[i]),len(val[i]),len(test[i]))
print("Types")
for i in range(4):
    print(len(train_set[i]),len(val_set[i]),len(test_set[i]))
    
