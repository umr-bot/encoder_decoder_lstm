# coding: utf-8
from collections import defaultdict
from tqdm import tqdm
from helper_functions import make_disjoint, write_folds
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

# unzip the tuples of norm and error tokens
train_norm,val_norm,test_norm=[[] for dummy in range(4)],[[] for dummy in range(4)],[[] for dummy in range(4)]
train_err,val_err,test_err=[[] for dummy in range(4)],[[] for dummy in range(4)],[[] for dummy in range(4)]
for i in range(4):
    train_norm[i],train_err[i] = list(zip(*[line.split(',') for line in train[i]]))
    val_norm[i],val_err[i] = list(zip(*[line.split(',') for line in val[i]]))
    test_norm[i],test_err[i] = list(zip(*[line.split(',') for line in test[i]]))

write_folds(train_norm,val_norm,test_norm,root_dir="folds/norm_foldset")
write_folds(train_err,val_err,test_err,root_dir="folds/foldset")

