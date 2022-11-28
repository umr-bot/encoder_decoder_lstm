# coding: utf-8
from collections import Counter, defaultdict
from tqdm import tqdm

def count_tup_errs(tups):
    err_cnt = 0
    for tok, norm_tok in tups:
        if tok!=norm_tok: err_cnt+=1

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def make_dictionary_folds(fn="./words_shuffled",out_dir="dictionary_folds/", return_flag=False, words=[]):
    """Make folds out of a dictionary file with a token on each line."""
    # if words list is not passed as argument load from file with name fn
    if words==[]:
        with open(fn) as f: words = [line.strip('\n') for line in f]
    ln = len(words)
    train = words[:int(ln*0.8)]
    val = words[int(ln*0.8):int(ln*0.9)]
    test = words[int(ln*0.9):]
    num_folds=4
    train_list = list(chunks(train,len(train)//num_folds))[0:num_folds]
    val_list = list(chunks(val,len(val)//num_folds))[0:num_folds]
    test_list = list(chunks(test,len(test)//num_folds))[0:num_folds]

    train_folds,val_folds,test_folds=[],[],[]
    for i in range(num_folds):
        with open(out_dir+"fold"+str(i+1)+"/train",'w') as f:
            train_folds.append(train_list[i])
            for word in train_list[i]: f.write(word+'\n')

        with open(out_dir+"fold"+str(i+1)+"/val",'w') as f:
            val_folds.append(val_list[i])
            for word in val_list[i]: f.write(word+'\n')

        with open(out_dir+"fold"+str(i+1)+"/test",'w') as f:
            test_folds.append(test_list[i])
            for word in test_list[i]: f.write(word+'\n')

    # if return flag argument was passed as True then return fold lists
    if return_flag: return train_folds,val_folds,test_folds

def read_dictionary_folds():
    # load dictionary values
    norm_train,norm_val,norm_test=[],[],[]
    for i in range(4):
        with open("dictionary_folds/fold"+str(i+1) +"/train") as f:
            norm_train.append([line.strip('\n') for line in f])
        with open("dictionary_folds/fold"+str(i+1) +"/val") as f:
            norm_val.append([line.strip('\n') for line in f])
        with open("dictionary_folds/fold"+str(i+1) +"/test") as f:
            norm_test.append([line.strip('\n') for line in f])
    return norm_train,norm_val,norm_test

def write_folds(folds,fn,root_dir="eng_folds/"):
    num_folds = len(folds)
    for i in range(num_folds):
        cnt=0
        keys,values=[],[]
        for k,v in folds[i].items():
            cnt+=len(v[0])
            keys.append(k)
            values.append(v)

        key_val_pairs = list(zip(keys, values))
        len_pairs = len(key_val_pairs)
        with open(root_dir+"fold"+str(i+1)+"/"+fn,'w') as f:
            for key,value in key_val_pairs:
                for tok in value[0]: f.write(key+','+tok+'\n')
        print(f"Number of tokens in {fn} fold_{i+1}: {cnt}")

# Confirm that no errors
#for i in range(4):
#    print(f"Train[{i}] errors {count_tup_errs(zip(train[i],norm_train[i]))}")
#    print(f"Val[{i}] errors {count_tup_errs(zip(val[i],norm_val[i]))}")
#    print(f"Test[{i}] errors {count_tup_errs(zip(test[i],norm_test[i]))}")
if __name__ == "__main__":
    num_folds = 4

    norm_train,norm_val,norm_test = read_dictionary_folds()

    with open("processed_missp/err_norm_data/err_norms_sorted.txt") as f:
        missp = [(tups.split(',')[0].strip(' '),tups.split(',')[1].strip(' \n')) for tups in f]
    errors = list(zip(*missp))[0]
    norms = list(zip(*missp))[1] # get all normalized tokens of missplelled words
    norm_err_dict = dict(zip(norms,errors)) # create zip of reversed order of errors and norms
    d = defaultdict(list)
    for tok,norm_tok in zip(errors,norms):
        d[norm_tok].append(tok)

    counts = Counter(norms) # bin normalized misspelled word tokens

    interval = len(missp)/(2*num_folds*3) # 2*num_folds*3 is from train, val and norm partitions
    cnt = 0
    ntrain=[defaultdict(list) for dummy in range(num_folds)]
    pop_list=[]
    for d_norm_tok,err_list in tqdm(d.items(), desc = "Looping over d"):
        for i in range(num_folds):
            for norm_tok in set(norm_train[i]):
                if norm_tok == d_norm_tok:
                    ntrain[i][norm_tok].append(err_list)
                    pop_list.append(norm_tok)
                    cnt+=len(err_list)
            if cnt > interval:
                cnt=0
                continue
    # so that validation does not consider items already reserved for train
    for item in pop_list:
        d.pop(item)
    cnt = 0
    nval=[defaultdict(list) for dummy in range(4)]
    pop_list=[]
    for d_norm_tok,err_list in tqdm(d.items(), desc = "Looping over d"):
        for i in range(num_folds):
            for norm_tok in set(norm_val[i]):
                if norm_tok == d_norm_tok:
                    nval[i][norm_tok].append(err_list)
                    pop_list.append(norm_tok)
                    cnt+=len(err_list)
            if cnt > interval:
                cnt=0
                continue
# so that test does not consider items already reserved for train and val
    for item in pop_list:
        d.pop(item)
    cnt = 0
    ntest=[defaultdict(list) for dummy in range(4)]
    pop_list=[]
    for d_norm_tok,err_list in tqdm(d.items(), desc = "Looping over d"):
        for i in range(num_folds):
            for norm_tok in set(norm_test[i]):
                if norm_tok == d_norm_tok:
                    ntest[i][norm_tok].append(err_list)
                    pop_list.append(norm_tok)
                    cnt+=len(err_list)
            if cnt > interval:
                cnt=0
                continue

    #write_folds(ntrain,fn="train")
    #write_folds(nval,fn="val")
    #write_folds(ntest,fn="test")
    with open("./words_shuffled") as f: words = set(line.strip('\n') for line in f)

    for i in tqdm(range(num_folds), desc="Finding set of words left over"):
        for norm_tok in ntrain[i].keys(): words.discard(norm_tok)
        for norm_tok in nval[i].keys(): words.discard(norm_tok)
        for norm_tok in ntest[i].keys(): words.discard(norm_tok)
    w_train,w_val,w_test = make_dictionary_folds(out_dir="words_shuffled_folds/", return_flag=True, words=list(words))
    # At this point ntrain,nval and ntest (denote this nX) contain only mispelling pairs
    # And w_train,w_val and w_test contain words types disjoint of the nX pairs

    # Append the dictionary words to the error folds
    for i in range(num_folds):
        for tok in w_train[i]: ntrain[i][tok].append([tok])
        for tok in w_val[i]: nval[i][tok].append([tok])
        for tok in w_test[i]: ntest[i][tok].append([tok])
    # Uncomment below if want ot write the final err_norm pairs out
    #write_folds(ntrain,fn="train")
    #write_folds(nval,fn="val")
    #write_folds(ntest,fn="test")

