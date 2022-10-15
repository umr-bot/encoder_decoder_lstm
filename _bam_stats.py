# coding: utf-8
from sort_error_types import DataHandler, distribute, split
from collections import defaultdict, Counter
from tqdm import tqdm

def rotate(l, n):
    return l[n:] + l[:n]
def flatten(l):
    return [item for sublist in l for item in sublist]
class BamStats:

    def __init__(self):
        self.data_handler = DataHandler()
        self.sorted_single_errs, self.err_counts = self.data_handler.get_err_counts_dicts()
        self.make_folds()
        self.make_err_folds()
        self.make_n_grams()
        self.bi_gram_folds = self.create_n_gram_folds(self.bi_grams)
        self.tri_gram_folds = self.create_n_gram_folds(self.tri_grams)
        self.write_n_grams(fold_tups=self.bi_gram_folds)
        self.write_n_grams(fold_tups=self.tri_gram_folds)


    def make_folds(self):
        """Make folds dict containing counts of error types in corpus"""
        fold_lens = sum(range(10)) + sum(range(11,19)) +21+23+26+29, sum([31,32,33,36,44,48,60]), sum([73,86,92]), 503
        self.num_folds = 4
        fold1 = [*range(1,5)]#+[*range(11,19)]#+[21,23,26,29]
        fold1_dev=[5]
        fold1_test=[6]
        fold2 = [*range(7,11)]
        fold2_dev=[11]
        fold2_test=[12,13]
        fold3 = [21,23,26,29,31,32,33,36]
        fold3_dev=[44]
        fold3_test=[48,60,73]
        fold4 = [*range(14,15)]+[503]
        fold4_dev=[15,86]
        fold4_test=[16,17,18,92]

        fold_lst = [fold1,fold2,fold3,fold4]
        fold_devs = [fold1_dev,fold2_dev,fold3_dev,fold4_dev]
        fold_tests = [fold1_test,fold2_test,fold3_test,fold4_test]
        folds = defaultdict(list)
        for i in range(self.num_folds):
            folds["fold"+str(i+1)].append(fold_lst[i])
            folds["fold"+str(i+1)+"dev"].append(fold_devs[i])
            folds["fold"+str(i+1)+"test"].append(fold_tests[i])
        self.folds=folds

    def make_err_folds(self, fold_check=False):
        self.err_folds=defaultdict(list)
        #err_folds=[[] for dummy in range(len(self.folds))]
        for single_err in self.sorted_single_errs:
            err_tok = single_err[0][2][1]
            err_count = self.err_counts[err_tok]
            for folds_key in self.folds.keys():
                if err_count in self.folds[folds_key][0]:
                    self.err_folds[folds_key].append(single_err)
                    #err_folds[fold_cnt].append(single_err)
        if fold_check:
            print("number of sentences with unequal sequence lengths vs total fold size per fold:")
            for fold_key in self.folds.keys():
                err_fold = self.err_folds[fold_key]
                y = [x for x in err_fold if len(x[0][0])!=len(x[0][1])]
                print(f"{fold_key}: {len(y)}, {len(err_fold)}")

            # Sanity check to see that there is no overlapping of types between folds
            # Note however that lists in x below are token lists and not type lists
            x = [[] for dummy in range(self.num_folds)]
            for i in range(self.num_folds):
                x[i] = [err_fold[0][2][1] for err_fold in self.err_folds["fold"+str(i+1)]]

            # All these statements should print True if all folds disjoint
            if set(tuple(flatten(x[1:4]))).isdisjoint(x[0]): print("fold1 disjoint of all other folds")
            if set(tuple(flatten(rotate(x,1)[1:4]))).isdisjoint(x[1]): print("fold2 disjoint of all other folds")

            if set(tuple(flatten(rotate(x,2)[1:4]))).isdisjoint(x[2]): print("fold3 disjoint of all other folds")

            if set(tuple(flatten(rotate(x,3)[1:4]))).isdisjoint(x[3]): print("fold4 disjoint of all other folds")

    def make_folds_for_enc_dec_model(self):
        """Input: DataHandler object data_handler,
                  number of folds num_folds
                  and defaultdict err_folds with key as fold name and values
                  a 4-tuple in the form of:
            ((errorful_sentece,normed_sentence,tuple_of_err_and_norm_word), frequency_count_of_norm_word)
        """
        ll = list(set([tok for toks in self.data_handler.without_err for tok in toks.split()]))
        interval = int(len(ll)/self.num_folds)
        ll_spliced=[]
        tups = defaultdict(set)
        for i in range(num_folds):
            if i == num_folds-1: ll_spliced = ll[i*interval:]
            else: ll_spliced = ll[i*interval:(i+1)*interval]
            # Add 250 tuples from correct tokens to dev and test
            dev = list(tok[0][2][1] for tok in self.err_folds["fold"+str(i+1)+"dev"])
            test = list(tok[0][2][1] for tok in self.err_folds["fold"+str(i+1)+"test"])
            dev_test = dev + test
            tups["fold"+str(i+1)+"dev"] = set([tup[0][2] for tup in self.err_folds["fold"+str(i+1)+"dev"]] + [(x,x) for x in ll_spliced[-1000:-500]])
            tups["fold"+str(i+1)+"test"] = set([tup[0][2] for tup in self.err_folds["fold"+str(i+1)+"test"]] + [(x,x) for x in ll_spliced[-500:]])
            tups["fold"+str(i+1)] = set([tup[0][2] for tup in self.err_folds["fold"+str(i+1)]])
            # note union does not do in place updates, hence have to assign result
            tups["fold"+str(i+1)]=tups["fold"+str(i+1)].union([(x,x) for x in ll_spliced[0:-1000] if x not in dev_test])
        return tups
    
    def write_tups(self,tups,out_fn="bam_folds/folds/"):
        for key_name in tups.keys():
            with open(out_fn+key_name, "w") as f:
                for tup in tups[key_name]: f.write(tup[0]+','+tup[1]+"\n")
    
    def make_n_grams(self):
        """Make data anagram hashing compatible"""
        # Assign bigram lists
        self.bi_grams,self.norm_bi_grams=[],[]
        for line,norm_line in zip(self.data_handler.without_err,self.data_handler.norm_without_err):
            toks,norm_toks = line.split(), norm_line.split()
            for i in range(len(toks)-1):
                self.bi_grams.append((toks[i],toks[i+1]))
                self.norm_bi_grams.append((norm_toks[i],norm_toks[i+1]))
        # Assign trigram lists
        self.tri_grams,self.norm_tri_grams=[],[]
        for line,norm_line in zip(self.data_handler.without_err,self.data_handler.norm_without_err):
            toks,norm_toks = line.split(), norm_line.split()
            for i in range(len(toks)-2):
                self.tri_grams.append((toks[i],toks[i+1],toks[i+2]))
                self.norm_tri_grams.append((norm_toks[i],norm_toks[i+1],norm_toks[i+2]))

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
        for i in range(len(fold_tups)):
            if len(fold_tups[i][0][0]) == 2:
                with open("bam_folds/n_grams/bi_grams_fold"+str(i+1),"w") as f:
                    for gram in fold_tups[i]: f.write(gram[0][0]+","+gram[0][1]+"\n")
            elif len(fold_tups[i][0][0]) == 3:
                with open("bam_folds/n_grams/tri_grams_fold"+str(i+1),"w") as f:
                    for gram in fold_tups[i]: f.write(gram[0][0]+','+gram[0][1]+','+gram[0][2]+"\n")

if __name__ == "__main__":
   bam_stats = BamStats()

#lines = [line for line in data_handler.without_err]
#lines_set = [set(line.split()) for line in data_handler.without_err]
#lines_tup = [(line,set(line.split())) for line in data_handler.without_err]
#l_fold1 = []
#tot_set = set()
#for line, line_set in lines_tup:
#    if line_set.isdisjoint(tot_set): l_fold1.append((line,line_set))
#    tot_set = tot_set.union(line_set)
#    
#lines_tup2 = []
#for line, line_set in lines_tup:
#    for l_fold, l_fold_set in l_fold1:
#        # use frozenset so that we can take set of the sets in lines_tup2
#        if line_set.isdisjoint(l_fold_set): lines_tup2.append((line, frozenset(line_set)))
#
#len(set(lines_tup2))
