# coding: utf-8
from collections import Counter
from eng_data.kfolds.balance_errors_in_dataset import Balancer
import itertools
def distribute(source_one, source_two, order=[]):
    step = (len(source_one) - 2)//(len(source_two) - 1)
    splice = source_one[1:-1] # grab all items except first one in source_one
    iters = [iter(splice)] * step # make tuple pairs of size 'step' out of 
                                  # spliced source_one eg. [a,b,c,d,e,f],step=2
                                  # then iters=[(a,b),(c,d),(e,f)]
    # iters = reorder(list(zip(*iters)))
    # flatten iters appended by source_one[0]
    compressed = itertools.chain(source_one[0:1], zip(*iters))
    unzipped = zip(compressed, source_two) 
    # eg. continued: if source_two = [0,1,2,3] then 
    # unzipped = [((source_one,0),(a,b),(c,d),(e,f)), (0,1,2,3)]
    flattened = list(flatten(unzipped)) # flatten unzipped into a 1xn dim list
    # then flattened= [(source_one[0],1), ((a,b),1), ((c,d),2), ((e,f),3)]
    # Finally, append source_one[-1] to flattened, another way is to return
    # list(itertools.chain(flattened, source_one[-1:]))
    return flattened.append(source_one[-1]) 

def flatten(lst):
    return sum(([x] if not isinstance(x, tuple) else flatten(x) for x in lst), [])

def reorder(lst,order):
    """Reorder a list according to another list of numbers"""
    tmp = list(zip(lst,order)) # pair lst elements with count number elements
    tmp.sort(key=lambda x: int(x[1]),reverse=True) # sort lst according 'order'
    return list(zip(*tmp))[0] # return only reordered lst list

bal = Balancer(original_fn="eng_data/kfolds/balanced_err_eng_50000",normed_fn="eng_data/kfolds/balanced_eng_50000")
with_err,norm_with_err,without_err,norm_without_err,err_tups = bal.check_err()

err = list(zip(with_err,norm_with_err))
#####################################################################
# Split sentences with multiple errors into single error sentences
single_errs=[]
j=0
for i in range(len(err)):
    sent, norm_sent = err[i][0], err[i][1]
    toks, ntoks = sent.split(), norm_sent.split()
    for tok_cnt, ntok_cnt in zip(range(len(toks)), range(len(ntoks))):
        tok, ntok = toks[tok_cnt], ntoks[ntok_cnt]
        if tok != ntok: # for every error token in sentence
            tmp = ntoks.copy()
            tmp[ntok_cnt] = toks[tok_cnt] # insert error into normed sentence
            tmp = " ".join([x for x in tmp])
            ntoks_sent = " ".join([x for x in ntoks])
            single_errs.append((tmp, ntoks_sent,(tok,ntok)))
            j+=1
            #print(f"tok_cnt:{tok_cnt}, tok:{tok}, ntok:{ntok}, j:{j}")
    #print(f"\nlen of toks:{len(toks)}")
#####################################################################
# Get counts of word frequencies
toks, ntoks = [],[]
for i in range(len(single_errs)):
    for tok, ntok in zip(single_errs[i][0].split(), single_errs[i][1].split()):
        if tok != ntok: 
            toks.append(tok)
            ntoks.append(ntok)

type_count = Counter(tok)
ntype_count = Counter(ntoks)
#####################################################################
# Assign the counts of the word errorr to the respective sentences
sorted_single_errs=[]
for single_err in single_errs:
    single_err_tup = single_err #tuple of sentence and its normed form
    single_err_cnt = ntype_count[single_err[2][1]]
    sorted_single_errs.append((single_err_tup, single_err_cnt))
sorted_single_errs.sort(key=lambda x: x[1], reverse=True)

cur_word=sorted_single_errs[0][0][2][1]
norm_cnt = 0
err_dict = dict()
err_counts = Counter()
for s in sorted_single_errs:
    norm_word = s[0][2][1]
#    if cur_word != norm_word:
#        err_dict[cur_word] = norm_cnt
#        cur_word = norm_word
#        norm_cnt += 0
#    else: 
#        err_counts[cur_word].update()
#        norm_cnt+=1
    err_counts.update([norm_word])

#err_sorted=[]
#for i in range(len(err)):
#    # iterate over sentences in err list
#    tok_count = 0
#    for tok, ntok in zip(err[i][0].split(), err[i][1].split()):
#        # iterate over tokens in sentence
#        if tok != ntok:
#            # ADD the count of a misspelled token to tok_count
#            tok_count += ntoks_count[ntok]
#
#    err_sorted.append((err[i], tok_count))
#    
#err_sorted.sort(key=lambda x: x[1], reverse=True)

#err_sorted_, counts = zip(*err_sorted) # TODO: include counts in sorting below
#errors,norm_errors = zip(*err_sorted_)
#sorted_list = []
#if len(without_err) > len(err_sorted): 
#    sorted_list = list(distribute(source_one=without_err,source_two=errors))
#    norm_sorted_list = list(distribute(source_one=norm_without_err,source_two=norm_errors))
#
#elif len(without_err) == len(err_sorted):
#    # Add empty sentences to source_one arguments for distribute function to
#    # work, source_one > source_two, hence we add empty sentence
#    sorted_list = distribute(source_one=without_err.append(''),source_two=errors)
#    norm_sorted_list = distribute(source_one=norm_without_err.append(''),source_two=norm_errors)
#
#else: # reverse order of arguments passed to distribute function
#      # as len(source_one) > len(source_two) for function to work
#    sorted_list = distribute(source_one=errors,source_two=without_err)
#    norm_sorted_list = distribute(source_one=norm_errors,source_two=norm_without_err)
#
