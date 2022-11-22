# coding: utf-8
"""Script written specificaly to load the data in root_dir/mispelling_files/ABODAT.643"""
with open("mispelling_files/ABODAT.643") as f:
    data = [line[:-2] for line in f]
X = []
for line in data:
    if '$' in line: continue
    tups = line.split(',')
    for toks in tups:
        tok = toks.split()
        X.append((tok[0],tok[1]))

