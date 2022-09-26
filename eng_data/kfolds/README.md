Folds are divided by following a even division of text until the final fold,
where if the text does not perfectly divide into even folds the last fold will
be the only fold with a different size.

Example we have a Bamabara corpora of approximately 27000 sequences of words,
which for 5-fold validation will result with each fold containining 5000
sequences except the final fold with will have a size of 7000 sequences.

To balance dataset use write\_balanced\_datasets function in balance\_errors\_in\_dataset.py file.
