# coding: utf-8
from tqdm import tqdm

with open("processed_missp/err_norm_data/err_norms_sorted.txt") as f:
    err_norms = [line.strip('\n').split(',') for line in f]
num_folds=5
folds=[]
for i in range(num_folds):
    train_val_test = []
    for grp in ["train","val","test"]:
        with open("trigrams/foldset"+str(i+1)+"/"+grp) as f:
            data = list(set(tuple(line.strip('\n').split()) for line in f))
        data = [list(elem) for elem in data] # conv lst of tup to lst of lst
        tri_err_norms = []
        for err_norm in tqdm(err_norms, desc=f"Fold {str(i+1)} {grp}"):
            for tri in data:
                err_tri = tri.copy()
                for tri_cnt in range(len(tri)):
                    if err_norm[1] == tri[tri_cnt]:
                        err_tri[tri_cnt] = err_norm[0]#replace norm with err
                        tri_err_norms.append((tri,err_tri))
        train_val_test.append(tri_err_norms)
        with open("trigram_norm_errs/norm_fold"+str(i+1)+"/"+grp,'w') as f:
            for tri_pair in tri_err_norms:
                f.write(tri_pair[0][0]+' '+tri_pair[0][1]+' '+tri_pair[0][2]+'\n')
        with open("trigram_norm_errs/fold"+str(i+1)+"/"+grp,'w') as f:
            for tri_pair in tri_err_norms:
                f.write(tri_pair[1][0]+' '+tri_pair[1][1]+' '+tri_pair[1][2]+'\n')
    folds.append(train_val_test)

