num_folds=5
train = [[] for i in range(num_folds)]
val, test = train.copy(), train.copy()
for i in range(num_folds):
    with open("fold"+str(i+1)+"/train") as f: train[i] = [line.strip('\n').split(',') for line in f]
    with open("fold"+str(i+1)+"/val") as f: val[i] = [line.strip('\n').split(',') for line in f]
    with open("fold"+str(i+1)+"/test") as f: test[i] = [line.strip('\n').split(',') for line in f]

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
