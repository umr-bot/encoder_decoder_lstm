# Usage of scripts in this directory:

## The extract_word_tuples_missp.py file can be used to reformat data files in this directory to generate bigram data for use in anagram hashing method

## make_folds.py, rorganize_tuple_folds.py and make_folds_disjoint can be used to format data for use in the encoder decoder model

## hand_checked_folds directory is the most clean folds that can be used and has been manuaaly cleaned from some irrelevent data from the folds directory

## make_train_from_val.py splits val data into train for enc_dec model. Anagram hashing trains on externally obtained bigrams and hence does not have train files in folds/foldset and fold/norm_foldset directories.

Call scripts in this order:
    0. extract_tuples_missp.py
    1. make_folds.py
    2. reorganize_tuple_folds.py
    3. make_train_from_val.py
    4. make_folds_disjoint.py
    5. shuffle_folds.sh
