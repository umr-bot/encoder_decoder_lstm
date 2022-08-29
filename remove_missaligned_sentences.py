"""Script to extract all lines where the number of words dont match
   between original and linguist corrected text.
"""
with open("data/folds/all") as f: all_ = [line for line in f]
with open("data/norm_folds/all") as f: nall = [line for line in f]

# variables to write out to files with
w_all,w_nall=[],[] # lines where line lens of original and linguist corrected text are the same
uneven_all, uneven_nall=[],[] # lines where line lens of original and linguist corrected text are different
for i in range(len(all_)):
    if len(all_[i])!=len(nall[i]):
        uneven_all.append(all_[i])
        uneven_nall.append(nall[i])
    w_all.append(all_[i])
    w_nall.append(nall[i])

with open("data/folds/all_removed_unequal",'w') as f:
    for line in w_all: f.write(line)
with open("data/norm_folds/all_removed_unequal",'w') as f:
    for line in w_all: f.write(line)
with open("data/folds/all_with_unequal",'w') as f:
    for line in uneven_all: f.write(line)
with open("data/norm_folds/all_with_unequal",'w') as f:
    for line in uneven_nall: f.write(line)

