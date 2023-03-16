num_folds=5
train = [[] for i in range(num_folds)]
val, test = train.copy(), train.copy()
for i in range(num_folds):
    with open("folds_with_overlap/fold"+str(i+1)+"/train") as f: train[i] = [line.strip('\n').split(',') for line in f]
    with open("folds_with_overlap/fold"+str(i+1)+"/val") as f: val[i] = [line.strip('\n').split(',') for line in f]
    with open("folds_with_overlap/fold"+str(i+1)+"/test") as f: test[i] = [line.strip('\n').split(',') for line in f]

def remove_overlap(overlapped_part,part1,part2):
    """ Remove overlap of norm tokens in err-norm pairs in overlapped_part
        with respect to part1 and part2 """
    val_tups = set(tup[1] for tup in part1)
    test_tups = set(tup[1] for tup in part2)

    train_copy=[]
    cnt = 0
    for train_tup in overlapped_part:
        if train_tup[1] not in val_tups and train_tup[1] not in test_tups: train_copy.append(train_tup)
        else: cnt+=1
    print(f"Overlap of {cnt} norm tokens")
    # uncomment line below for sanity check to make sure overlap is zero
    #print( len( set(tup[1] for tup in train_copy).intersection(val_tups) ))
    return train_copy

for i in range(num_folds):
    print(f"Cleaning train fold {i+1}")
    train[i] = remove_overlap(train[i],val[i],test[i])
    train[i] = list(set(tuple(tup) for tup in train[i]))
    print(f"Cleaning val fold {i+1}")
    val[i] = remove_overlap(val[i],train[i],test[i])
    val[i] = list(set(tuple(tup) for tup in val[i]))
    print(f"Cleaning test fold {i+1}")
    test[i] = remove_overlap(test[i],val[i],train[i])
    test[i] = list(set(tuple(tup) for tup in test[i]))
    print("------------------------------------------")

# tup is in form error-norm
for grp,name in zip([train, val, test], ["train","val","test"]):
    for i in range(num_folds):
        cnt=0
        for tup in grp[i]:
            if tup[0] != tup[1]: cnt+=1
        print(f"num {name} errors: {cnt}")
    print("")
print("---------------------------------------------------------")
for grp,name in zip([train, val, test], ["train","val","test"]):
    for i in range(num_folds): print(f"num {name} tokens {len(grp[i])}")
    print("         -----")
    for i in range(num_folds):
        # get types from set of norm_tokens
        print(f"num {name} types {len(set(tup[1] for tup in grp[i]))}")
    print("")

for i in range(num_folds):
    with open("fold"+str(i+1)+"/train",'w') as f:
        for tup in train[i]: f.write(tup[0]+','+tup[1] + '\n')
    with open("fold"+str(i+1)+"/val",'w') as f:
        for tup in val[i]: f.write(tup[0]+','+tup[1] + '\n')
    with open("fold"+str(i+1)+"/test",'w') as f:
        for tup in test[i]: f.write(tup[0]+','+tup[1] + '\n')

