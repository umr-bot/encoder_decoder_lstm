# coding: utf-8
from collections import Counter
from eng_data.kfolds.balance_errors_in_dataset import Balancer
import itertools
def distribute(source_one, source_two):
    step = (len(source_one) - 2)//(len(source_two) - 1)
    splice = source_one[1:-1]
    iters = [iter(splice)] * step
    compressed = itertools.chain(source_one[0:1], zip(*iters))
    unzipped = zip(compressed, source_two)
    flattened = flatten(unzipped)
    return itertools.chain(flattened, source_one[-1:])
def flatten(lst):
    return sum(([x] if not isinstance(x, tuple) else flatten(x) for x in lst), [])

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

err_sorted_, counts = zip(*err_sorted) # TODO: include counts in sorting below
errors,norm_errors = zip(*err_sorted_)
sorted_list = []
if len(without_err) > len(err_sorted): 
    sorted_list = list(distribute(source_one=without_err,source_two=errors))
    norm_sorted_list = list(distribute(source_one=norm_without_err,source_two=norm_errors))

elif len(without_err) == len(err_sorted):
    # Add empty sentences to source_one arguments for distribute function to
    # work, source_one > source_two, hence we add empty sentence
    sorted_list = list(distribute(source_one=without_err.append(''),source_two=errors))
    norm_sorted_list = list(distribute(source_one=norm_without_err.append(''),source_two=norm_errors))

else: # reverse order of arguments passed to distribute function
      # as len(source_one) > len(source_two) for function to work
    sorted_list = list(distribute(source_one=errors,source_two=without_err))
    norm_sorted_list = list(distribute(source_one=norm_errors,source_two=norm_without_err))

