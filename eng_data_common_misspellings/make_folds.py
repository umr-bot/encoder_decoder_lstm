""" Create approximately even split and balanced folds of n-gram data.
    Created: 16 Oct 2022
    Author: Umr Barends"""
from collections import Counter
import random
from tqdm import tqdm
def get_non_error_bigrams():
    tuples,cnts=[],[]
    with open("bigrams.txt") as f:
        for line in f:
            toks=line.split()
            if toks[0]!="0BEGIN.0" and toks[1]!="0END.0":
                tuples.append((toks[0],toks[1]))
                cnts.append(toks[2].rstrip('\n'))
    cnts = [int(c) for c in cnts] # convert from str numeral to int
    assert(len(tuples)==len(cnts))
    tups=[]
    for i in tqdm(range(len(tuples))): # convert from word type to token list
        for j in range(cnts[i]):
            tups.append(tuples[i])
    assert(len(tuples)==len(cnts))
    return tups, cnts

class MakeFolds:
    def __init__(self,n_gram_fn='', bigram_flag=True):
        self.bigram_flag=bigram_flag
        if bigram_flag==False: #use read in file name
            with open(n_gram_fn) as f:
                self.bi_grams=[(line.split(',')[0].lstrip(' '),line.split(',')[1].rstrip('\n')) for line in f]
        else: self.bi_grams,self.cnts=get_non_error_bigrams()
        self.bi_grams=random.sample(self.bi_grams,k=len(self.bi_grams))
        self.fold_tups=self.create_n_gram_folds(self.bi_grams)
        self.write_n_grams(self.fold_tups)

    def create_n_gram_folds(self,n_grams):
        # Split n-grams into folds
        n_grams_cnts = Counter(n_grams)
        tot_cnt=0
        for k,v in n_grams_cnts.most_common():
            tot_cnt+=v
        partition_size=int(tot_cnt/16)
        part_tups=[]
        tups=[]
        temp_cnt=0
        for k,v in n_grams_cnts.most_common():
            temp_cnt+=v
            tups.append((k,v))
            if temp_cnt>=partition_size or temp_cnt>partition_size-100:
                part_tups.append(tups)
                tups=[]
                temp_cnt=0
        # Assign folds
        z=[(x,y) for x, y in zip(part_tups[:(len(part_tups)+1)//2], reversed(part_tups))]
        fold_tups=[]
        for i in range(0,8,2):
            l1=[y for x in z[i] for y in x]
            l2=[y for x in z[i+1] for y in x]
            fold_tups.append(l1+l2)
        # Sanity check
        t=0
        for i in range(4):
            for x in fold_tups[i]:
                t+=x[1]
            print(t)
            t=0
        return fold_tups

    def write_n_grams(self, fold_tups): 
        """Write out data created in n_gram folds"""
        root_dir=''
        if self.bigram_flag==False: root_dir="err_tuples/"
        else: root_dir="n_grams/"
        for i in tqdm(range(len(fold_tups)),desc=f"Writing folds"):
            if len(fold_tups[i][0][0]) == 2:
                with open(root_dir+"bi_grams_fold"+str(i+1),"w") as f:
                    for gram in fold_tups[i]: f.write(gram[0][0]+","+gram[0][1]+"\n")
            elif len(fold_tups[i][0][0]) == 3:
                with open(root_dir+"tri_grams_fold"+str(i+1),"w") as f:
                    for gram in fold_tups[i]: f.write(gram[0][0]+','+gram[0][1]+','+gram[0][2]+"\n")

if __name__ == "__main__":
    makefolds=MakeFolds("wiki_missp_aspell.txt",bigram_flag=False)
    #makefolds=MakeFolds(bigram_flag=True)
