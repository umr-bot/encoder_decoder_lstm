# coding: utf-8
import re
def extract_missp_dat(fn):
    with open(fn) as f:
        sentences=[line.rstrip('\n') for line in f]
    lines=" ".join(line for line in sentences)
    words=re.findall(r'\$(.*)\$', lines)
    #words.insert(0,'$')
    words=[tok for toks in words for tok in toks.split()]
    words[0] = '$'+words[0]
    words.append('$')
    tuples=[]
    #for toks in words:
    correct,incorrect='',[] # one to many mapping of spellins
    tups=[]
    for tok in words: #last character empty literal
        if tok[0]=='$':
            #tuples.append((correct,incorrect))
            correct = tok[1:]
            #incorrect=[]
        else: tuples.append((correct,tok))#incorrect.append(tok)
    tuples=tuples[1:] # discard empty literal and list tuple
    return tuples
def extract_wiki():
    data=[]
    with open("wikipedia_miss.txt") as f:
        for line in f:
            correct=line.split("->")[0]
            incorrect=line.split("->")[1].rstrip('\n')
            incorrect_toks = incorrect.split(',') # one to many mapping
            # for every token
            for i in range(len(incorrect_toks)): data.append((correct, incorrect_toks[i]))

    return data

missp_tuples=extract_missp_dat("missp.dat")
aspell_tuples=extract_missp_dat("aspell.dat")
wiki_data=extract_wiki()
data = missp_tuples+wiki_data+aspell_tuples
with open("wiki_missp_aspell.txt",'w') as f:
    for tup in data: f.write(f"{tup[1]},{tup[0]}\n")

