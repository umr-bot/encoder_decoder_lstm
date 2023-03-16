# coding: utf-8
from collections import Counter

# get all tokens from original transcript
with open("trigram_errors") as f:
    text=[]
    for line in f:
        text += [tok for tok in line.strip('\n').split()]
#########################################

# Get frequncy counts of how many times a trigram occurs
word_cnts = Counter(text)
parts=[[],[],[],[],[]]
for i in range(5):
    with open("folds/part"+str(i+1)) as f:
        parts[i] = [tuple(line.strip('\n').split(','))[1] for line in f]
        parts[i] = list(set(parts[i]))
######################################################
# Print fold counts out to screen
print("Printing fold counts")
for i in range(5):
    cnt=0
    for tok in parts[i]: cnt += word_cnts[tok]
    print(f"Number of toks in fold{i+1}: {cnt}",end="   ")
    print(f"Number of types in fold{i+1}: {len(parts[i])}")
print("----------------------------------------------------")
#####################################################
# Get number of counts in each foldset and print to screen
foldset_norms=[[],[],[],[],[]]
for name in ["train","val","test"]:
    print(f"Stats for {name}")
    for i in range(5):
        with open("folds/fold"+str(i+1)+"/"+name) as f:
            foldset_norms[i] = [line.strip('\n').split(',')[1] for line in f]
        cnt=0
        for tok in foldset_norms[i]: cnt += word_cnts[tok]
        print(f"Number of toks in foldset{i+1}: {cnt}",end="   ")
        print(f"Number of types in foldset{i+1}: {len(foldset_norms[i])}")

