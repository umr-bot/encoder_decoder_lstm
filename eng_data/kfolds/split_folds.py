"""Script to split some corpus into k-folds
   Date: Aug 11 2022 
   Author: Umr Barends
"""
import string

def split_corpus(corpus_fn="eng_50000",output_dir="folds/"):
    corpus_text_file = corpus_fn # file containg entire corpus to split
    with open(corpus_text_file) as f:
        raw_data = [seq for seq in f]
    
    data = []
    for text in raw_data:
        new_text = ''.join(c for c in text if c not in string.punctuation)
        data.append(new_text)

    k_folds = 5
    fold_len = int(len(data)/k_folds)

    fold, folds = [],[]
    for i in range(len(data)):
        fold.append(data[i])
        if (i+1) % fold_len == 0:
            folds.append(fold)
            fold = []

    for i in range(k_folds):
        with open(output_dir+"fold"+str(i+1),"w") as f:
            for seq in folds[i]:
                f.write(seq)
