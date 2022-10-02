# coding: utf-8
from collections import Counter
from eng_data.kfolds.balance_errors_in_dataset import Balancer

bal = Balancer(original_fn="eng_data/kfolds/balanced_err_eng_50000",normed_fn="eng_data/kfolds/balanced_eng_50000")
with_err,norm_with_err,without_err,norm_without_err,err_tups = bal.check_err()

err = list(zip(with_err,norm_with_err))

toks, ntoks = [],[]
for i in range(len(err)):
    for tok, ntok in zip(err[i][0].split(), err[i][1].split()):
        if tok != ntok: 
            toks.append(tok)
            ntoks.append(ntok)

ntoks_count = Counter(ntoks)
toks_count = Counter(ntok)

err_sorted=[]
for i in range(len(err)):
    # iterate over sentences in err list
    tok_count = 0
    for tok, ntok in zip(err[i][0].split(), err[i][1].split()):
        # iterate over tokens in sentence
        if tok != ntok:
            # ADD the count of a misspelled token to tok_count
            tok_count += ntoks_count[ntok]

    err_sorted.append((err[i], tok_count))
    
err_sorted.sort(key=lambda x: x[1], reverse=True)
