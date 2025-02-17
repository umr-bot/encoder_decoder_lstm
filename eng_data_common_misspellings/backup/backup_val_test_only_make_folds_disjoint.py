# coding: utf-8
""" Script that makes the folds in foldsets in folds/ directory generated by 
    make_folds.py disjoint/independent. The function make_folds_disjoint 
    reorders a list k folds with 2-tuple elements such that the k folds have
    elements that are unique across folds. 
"""
from collections import Counter
from tqdm import tqdm
class MakeFoldsDisjoint:

    def __init__(self,root_dir='',norm_root_dir='',out_dir='',norm_out_dir=''):
        self.load_folds(root_dir,norm_root_dir)
        self.norm_val_sets = self.make_foldsets_disjoint(self.norm_val_sets)
        self.norm_test_sets = self.make_foldsets_disjoint(self.norm_test_sets)
        # Write type lists into files in file path "out_dir+fold_num+suffix"
        self.write_folds(self.norm_val_sets,out_dir=norm_out_dir,suffix="val_types")
        self.write_folds(self.norm_test_sets,out_dir=norm_out_dir,suffix="test_types")
        self.val_folds,self.test_folds,self.norm_val_folds,self.norm_test_folds=self.convert_fold_types_to_tokens()
        # Write type lists into files in file path "out_dir+fold_num+suffix"
        self.write_folds(self.val_folds,out_dir=out_dir,suffix="val")
        self.write_folds(self.norm_val_folds,out_dir=norm_out_dir,suffix="val")
        self.write_folds(self.test_folds,out_dir=out_dir,suffix="test")
        self.write_folds(self.norm_test_folds,out_dir=norm_out_dir,suffix="test")

    def load_folds(self, root_dir='',norm_root_dir=''):
        """root_dir is folder name prefix where folds are stored."""
        self.val,self.test,self.norm_val,self.norm_test=[],[],[],[]
        self.val_sets,self.test_sets,self.norm_val_sets,self.norm_test_sets=[],[],[],[]
        self.val_tups,self.test_tups=[],[]
        for i in range(4):
            #TODO: extract set from Counter object
            with open(root_dir+str(i+1)+"/val") as f: val=[line.rstrip('\n') for line in f]
            self.val.append(val)
            self.val_sets.append(set(val))
            with open(root_dir+str(i+1)+"/test") as f: test=[line.rstrip('\n') for line in f]
            self.test.append(test)
            self.test_sets.append(set(test))
            with open(norm_root_dir+str(i+1)+"/val") as f: norm_val=[line.rstrip('\n') for line in f]
            self.norm_val.append(norm_val)
            self.norm_val_sets.append(set(norm_val))
            with open(norm_root_dir+str(i+1)+"/test") as f: norm_test=[line.rstrip('\n') for line in f]
            self.norm_test.append(norm_test)
            self.norm_test_sets.append(set(norm_test))
            # make (incorrect_tok,correct_tok) tuples of val and test data
            self.val_tups+=list(zip(val,norm_val))
            self.test_tups+=list(zip(test,norm_test))

    def make_foldsets_disjoint(self, norm_folds):
        """ Function that reorders a list k folds with 2-tuple elements
            such that the k folds have elements that are unique across folds """
        for i in range(len(norm_folds)):
            for j in range(len(norm_folds)):
                if j != i:
                    norm_folds[j] = norm_folds[j]-norm_folds[i].intersection(norm_folds[j])
        return norm_folds

    def check_fold_dependencies(self):
        for i in range(len(norm_folds)):
            print(f"Checking fold {i+1} dependencies")
            for j in range(len(norm_folds)):
                if j != i:
                    disjoint=norm_folds[i].isdisjoint(norm_folds[j])
                    if disjoint:print(f"Fold {i+1} is disjoint of fold {j+1}")
            print("----------------------------")
    
    def convert_fold_types_to_tokens(self,num_folds=4):
        val_folds,test_folds,norm_val_folds,norm_test_folds=[],[],[],[]
        for i in tqdm(range(num_folds),desc="Converting folds"):
            val_token_fold, norm_val_token_fold = self.convert_types_to_tokens(self.val_tups,self.val_sets[i],fdesc=(i+1,'val_'))
            val_folds.append(val_token_fold)
            norm_val_folds.append(norm_val_token_fold)
            test_token_fold, norm_test_token_fold =self.convert_types_to_tokens(self.val_tups,self.test_sets[i],fdesc=(i+1,'test_'))
            test_folds.append(test_token_fold)
            norm_test_folds.append(norm_test_token_fold)
        return val_folds,test_folds,norm_val_folds,norm_test_folds
    def convert_types_to_tokens(self, tups, types,fdesc=[]):
        """Convert types fold into token fold"""
        fold_num,norm_str=fdesc[0],fdesc[1]
        token_fold,norm_token_fold=[],[]
        for typ in tqdm(types,desc=f"Converting {norm_str}fold{fold_num} types to tokens"):
            for tup in tups:
                if typ==tup[1]:
                    token_fold.append(tup[0])
                    norm_token_fold.append(typ)
                    #print(tup[0],typ)
        return token_fold,norm_token_fold

    def write_folds(self,folds,out_dir='',suffix=''):
        for i in range(len(folds)):
            with open(out_dir+str(i+1)+"/"+suffix,'w') as f:
                for tok in folds[i]: f.write(tok+'\n')

if __name__ == "__main__":
    make_dis = MakeFoldsDisjoint(root_dir="unbalanced_folds/foldset",norm_root_dir="unbalanced_folds/norm_foldset",out_dir="folds/foldset",norm_out_dir="folds/norm_foldset")
