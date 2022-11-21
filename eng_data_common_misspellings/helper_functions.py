# coding: utf-8
from tqdm import tqdm
def write_folds(train,val,test,root_dir):
    """Write out folds containing tuples of tokens on each line"""
    for i in range(4):
        with open(root_dir+str(i+1)+"/train_"+str(i+1), 'w') as f:
            for line in tqdm(train[i], desc=f"train fold {i+1}"):
                f.write(line+'\n')
        with open(root_dir+str(i+1)+"/val_"+str(i+1), 'w') as f:
            for line in tqdm(val[i], desc=f"val fold {i+1}"):
                f.write(line+'\n')
        with open(root_dir+str(i+1)+"/test_"+str(i+1), 'w') as f:
            for line in tqdm(test[i], desc=f"test fold {i+1}"):
                f.write(line+'\n')

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
