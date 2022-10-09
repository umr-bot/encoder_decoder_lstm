# coding: utf-8
from collections import Counter
from artificial_corrupted_eng_data.kfolds.balance_errors_in_dataset import Balancer
import itertools
import string

def split(raw_data, k_folds=5):
    data=[]
    for text in raw_data:
        new_text = ''.join(c for c in text if c not in string.punctuation)
        data.append(new_text)
    fold_len = int(len(data)/k_folds)
    fold, folds = [],[]
    for i in range(len(data)):
        fold.append(data[i])
        if (i+1) % fold_len == 0:
            folds.append(fold)
            fold = []
    return folds

def distribute(source_one, source_two, order=[]):
    step = (len(source_one) - 2)//(len(source_two) - 1)
    splice = source_one[1:-1] # grab all items except first one in source_one
    iters = [iter(splice)] * step # make tuple pairs of size 'step' out of 
                                  # spliced source_one eg. [a,b,c,d,e,f],step=2
                                  # then iters=[(a,b),(c,d),(e,f)]
    # iters = reorder(list(zip(*iters)))
    # flatten iters appended by source_one[0]
    compressed = itertools.chain(source_one[0:1], zip(*iters))
    unzipped = list(zip(compressed, source_two))
    # eg. continued: if source_two = [0,1,2,3] then 
    # unzipped = [((source_one,0),(a,b),(c,d),(e,f)), (0,1,2,3)]
    flattened = list(flatten(unzipped)) # flatten unzipped into a 1xn dim list
    # then flattened= [(source_one[0],1), ((a,b),1), ((c,d),2), ((e,f),3)]
    # Finally, append source_one[-1] to flattened, another way is to return
    # list(itertools.chain(flattened, source_one[-1:]))
    flattened.append(source_one[-1]) 
    return flattened

def flatten(lst):
    return sum(([x] if not isinstance(x, tuple) else flatten(x) for x in lst), [])

def reorder(lst,order):
    """Reorder a list according to another list of numbers"""
    tmp = list(zip(lst,order)) # pair lst elements with count number elements
    tmp.sort(key=lambda x: int(x[1]),reverse=True) # sort lst according 'order'
    return list(zip(*tmp))[0] # return only reordered lst list

class DataHandler:

    def __init__(self,original_fn="data/balanced_all",normed_fn="data/balanced_norm_all"):
        #bal = Balancer(original_fn="eng_data/kfolds/balanced_err_eng_50000",normed_fn="eng_data/kfolds/balanced_eng_50000")
        bal = Balancer(original_fn=original_fn,normed_fn=normed_fn)
        with_err,norm_with_err,without_err,norm_without_err,err_tups = bal.check_err()
        self.without_err,self.norm_without_err=without_err,norm_without_err
        self.sorted_single_errs,self.err_types,self.err_counts = self.create_counts_dicts(with_err, norm_with_err)
        #self.sorted_single_,self.types,self.counts = self.create_counts_dicts(without_err, norm_without_err)
    
    def get_err_counts_dicts(self):
        return self.sorted_single_errs,self.err_counts

    def create_counts_dicts(self,data,norm_data):
        err = list(zip(data,norm_data))
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
        # Assign the counts of the word error to the respective sentences
        sorted_single_errs=[]
        for single_err in single_errs:
            single_err_tup = single_err #tuple of sentence and its normed form
            single_err_cnt = ntype_count[single_err[2][1]]
            sorted_single_errs.append((single_err_tup, single_err_cnt))
        sorted_single_errs.sort(key=lambda x: x[1], reverse=True)

        #####################################################################
        # Get number of word tokens associated with types in ntype_count
        err_counts = Counter()
        for s in sorted_single_errs:
            norm_word = s[0][2][1] # normed word token associated with word type
            err_counts.update([norm_word])
        err_types, err_type_count = zip(*err_counts.items())

        return sorted_single_errs,err_types,err_counts

    def filter_word_types(self):
        # Filter without err according to err types
        pass
