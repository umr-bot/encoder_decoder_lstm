# coding: utf-8
"""Wrapper file for balanceing erros in corpus and splitting it."""

from balance_errors_in_dataset import Balancer
from split_folds import split_corpus
from check_fold_errors import print_num_errs

bal = Balancer(original_fn="source_files/err_eng_50000",normed_fn="source_files/eng_50000")

out_fn="balanced_err_eng_50000"
out_norm_fn="balanced_eng_50000"

bal.write_balanced_datasets(out_fn=out_fn,out_norm_fn=out_norm_fn,k=5)

split_corpus(corpus_fn=out_fn,output_dir="folds/")
split_corpus(corpus_fn=out_norm_fn,output_dir="norm_folds/")

print_num_errs(bal=bal)
