# coding: utf-8
val_folds,norm_val_folds=[],[]
test_folds,norm_test_folds=[],[]
for i in range(4):
    with open("err_tuples/bi_grams_fold"+str(i+1)) as f: train=[tup.split(',') for tup in f]
    val=[]
    norm_val=[]
    for x in train:
        val.append(x[0])
        norm_val.append((x[1].strip('\n')))
    val_folds.append(val[:len(val)-500])
    norm_val_folds.append(norm_val[:len(norm_val)-500])
    test_folds.append(val[-500:])
    norm_test_folds.append(norm_val[-500:])
for i in range(len(val_folds)):
    val_fold,test_fold = val_folds[i],test_folds[i]
    with open("unbalanced_folds/foldset"+str(i+1)+"/val",'w') as f:
        for line in val_fold: f.write(line+'\n')
    with open("unbalanced_folds/foldset"+str(i+1)+"/test",'w') as f:
        for line in test_fold: f.write(line+'\n')
    norm_val_fold,norm_test_fold = norm_val_folds[i],norm_test_folds[i]
    with open("unbalanced_folds/norm_foldset"+str(i+1)+"/val",'w') as f:
        for line in norm_val_fold: f.write(line+'\n')
    with open("unbalanced_folds/norm_foldset"+str(i+1)+"/test",'w') as f:
        for line in norm_test_fold: f.write(line+'\n')
 
