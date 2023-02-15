# coding: utf-8
from tqdm import tqdm
import nltk
from collections import Counter

def multi_replace(text):
    chars = '\\`/*_{}[]()>“”\"#+-.,!$\n'
    for c in chars: text = text.replace(c, '')
    return text

for i in range(4):
    # Read in data to be split into trigrams
    with open("norm_fold"+str(i+1)+"/train_sents") as f:
        data = [multi_replace(line).lower() for line in f]
    ngrams = []
    for line in tqdm(data, desc="Looping over lines"):
        toks = line.split()
        grams = [gram for gram in nltk.ngrams(toks, 3)]
        for gram in grams: ngrams.append(gram)
    ngram_types = Counter(ngrams)
    print(len(ngrams),len(ngram_types))
    # write out trigram data
    with open("fold"+str(i+1)+"/train",'w') as f:
        for l in list(ngram_types): f.write(l[0]+' '+l[1]+' '+l[2]+'\n')
