# coding: utf-8
from collections import defaultdict
interval = len(missp)/8
cnt = 0
nt0 = defaultdict(list)
for d_norm_tok,err_list in d.items():
    for tok, norm_tok in zip(train[0], norm_train[0]):
        if norm_tok == d_norm_tok:
            nt0[norm_tok].append(err_list)
            cnt+=len(err_list)
    if cnt > interval: break
