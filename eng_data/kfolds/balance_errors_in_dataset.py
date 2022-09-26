# coding: utf-8
"""Script to balance orthography (spelling) errors across  a dataset for 5-fold   k-fold cross validation but not to split it into folds.
"""
import math
from random import shuffle

class Balancer:

    def __init__(self,original_fn, normed_fn):
        self.original_fn = original_fn
        self.normed_fn = normed_fn

    def check_err(self,input_fn="",normed_input_fn=""):
        """Function that extracts sentences with and without errors, with
           additional variable err_tups that contains triplet tuples
           with structure (original_word,normed_word,sentence_index).
           Use input_fn and normed_input_fn to check specific files
           for error differences.
        """
        if input_fn == "": all_fn, norm_fn = self.original_fn, self.normed_fn
        else: all_fn, norm_fn = input_fn, normed_input_fn

        # assign original unbalanced corpus
        with open(all_fn) as f: all_ = [line.strip('\n') for line in f]
        # assign original unbalanced linguist corrected corpus
        with open(norm_fn) as f: norm_all = [line.strip('\n') for line in f]
        num_err = 0
        with_err=[] # lines with errors in original file
        norm_with_err=[] # lines with errors in normed file
        without_err = [] # lines without errors in original file
        norm_without_err =[] # lines without errors in normed file
        err_tups = [] # error and correction pairs
        for i in range(len(all_)):
            toks,norm_toks=all_[i].split(), norm_all[i].split()
            if toks != norm_toks:
                for tok, n_tok in zip (toks, norm_toks):
                    if tok != n_tok:
                        num_err+=1
                        # err_tup contains a tuple of original token,error_token,line index on which error occurs
                        err_tups.append((tok, n_tok,i))
                with_err.append(all_[i]),norm_with_err.append(norm_all[i])
            else:
                without_err.append(all_[i])
                norm_without_err.append(norm_all[i])
        return with_err,norm_with_err,without_err,norm_without_err,err_tups

    def get_word_types_and_tokens(self,k=5,print_flag=False):
        """Get word types and tokens of folds"""
        err_tup_list=[]
        for i in range(1,k+1):
            with_err,norm_with_err,without_err,norm_without_err,err_tups = self.check_err(original_fn="folds/fold"+str(i), normed_fn="norm_folds/fold"+str(i))
            err_tup_list.append(err_tups)
        if print_flag:
            for i in range(k):
                # split err_tup_list according to indices 0,1,2
                # with indices mapped 0=original_word,1=norm_word,2=line_index
                e_columns = list(zip(*err_tup_list[i]))
                print(len(set(e_columns[0]))) # print word type errors
        return err_tup_list

    def get_word_type_errors_per_fold(self,fold=1):
        """Input fold number with first fold with starting index 1
           Return list of word type errors in a fold.
           Indices 'ind' maps to the variable err_tups returned by
           check_err function."""
        err_tup_list = self.get_word_types_and_tokens()
        ind=[] 
        for i in range(len(err_tup_list[fold-1])-1):
            if err_tup_list[fold-1][i] == err_tup_list[fold-1][i+1]: ind.append(i)
        return ind

    def get_word_type_errors_all_folds(self,k=5, print_flag=False):
        """Get word type errors in each fold of dataset.
           Contents in list of indices 'inds' maps to the variable err_tups returned by
           check_err function."""
        inds=[]
        for i in range(1,k+1):
            inds.append(self.get_word_type_errors_per_fold(fold=i))
        if print_flag:
            for j in range(len(inds)):
                print(len(inds[j]))
        return inds

    def write_balanced_datasets(self,out_fn,out_norm_fn,k=5):
        """ Input:  unbalanced corpus as a list of sentences
                    unbalanced normed corpus as list of sentences
                    name of balanced corpus file to write to
                    name of balanced normed corpus to write to

            Do:     write out files with the names given as arguments to def
            Return: nothing"""
        with_err,norm_with_err,without_err,norm_without_err,err_tups = self.check_err()
        num_errs = len(err_tups)
        # truncate lists to be divisible by k for k-folds validation
        without_err = without_err[0:int(len(without_err)/k)*k]
        norm_without_err = norm_without_err[0:int(len(without_err)/k)*k]

        # Randomly shuffle lines containing errors to balance the per line
        # error rate for lines with errors
        perm = list(range(len(with_err)))
        shuffle(perm)
        with_err = [with_err[index] for index in perm]
        norm_with_err = [norm_with_err[index] for index in perm]

        # define length of array to append at every nth index of without_err
        if len(without_err) > len(with_err):
            # TODO: Have not tested whether this case works as intended yet
            #       but beleive it should work. So in case errors are found, look here.
            nth_skip = int(len(without_err)/len(with_err)) # less sentences with errors than ones without
            j=0
        else:
            nth_skip = int(len(with_err)/len(without_err)) # more sentences with errors than ones without
            i,j = 0,0
            while i < len(without_err):
                # Insert, evenly spread error lines into every nth line into without_err lists
                [without_err.insert(i, element) for element in with_err[j:j+nth_skip]]
                [norm_without_err.insert(i, element) for element in norm_with_err[j:j+nth_skip]]
                j+=nth_skip
                i += nth_skip + 1

        # Add leftover lines with errors to beginning of output lists
        left_overs = len(with_err)-j
        nth_skip = int(len(without_err)/left_overs)
        k = 0
        while k < left_overs:
            #[without_err.insert(k,element) for element in with_err[j:]]
            #[norm_without_err.insert(k,element) for element in norm_with_err[j:]]
            without_err.insert(k,with_err[j])
            norm_without_err.insert(k,norm_with_err[j])
            j+=1
            k+= nth_skip+1
        # truncate lists to get final balanced output
        # theoretically only up to k sequences/senetences should be truncated
        # where k is the number of folds that will be used in cross-validation
        k = 5
        trunc = len(norm_without_err) - int(len(norm_without_err)/k)*k
        out = without_err[0:-trunc]
        norm_out = norm_without_err[0:-trunc]
        #out,norm_out = without_err,norm_without_err

        # ouput balanced data to files
        with open(out_fn,'w') as f:
            for line in out:
                for tok in line:
                    f.write(tok)
                f.write('\n')
        with open(out_norm_fn,'w') as f:
            for line in norm_out:
                for tok in line:
                    f.write(tok)
                f.write('\n')

# __name__ is assigned a srting depending on where script is called from
# if called directly then __name__ = "__main__"
# else __name__ = "name_of_script_this_script_is_being_called_from"
if __name__ == "__main__":
    balancer = Balancer(original_fn="source_files/err_eng_50000",normed_fn="source_files/eng_50000")
    inds = balancer.get_word_type_errors_all_folds(print_flag=True)


