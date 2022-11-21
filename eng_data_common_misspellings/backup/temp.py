# coding: utf-8
for i in range(4):
    cnt=0
    keys,values=[],[]
    for k,v in ntrain[i].items():
        cnt+=len(v[0])
        keys.append(k)
        values.append(v)

    key_val_pairs = list(zip(keys, values))
    len_pairs = len(key_val_pairs)
    
    with open("missp_folds/train_"+str(i+1),'w') as f:
        for key,value in key_val_pairs[:len_pairs//3]:
            for tok in value[0]: f.write(key+','+tok+'\n')
    with open("missp_folds/val_"+str(i+1),'w') as f:
        for key,value in key_val_pairs[len_pairs//3:]:
            for tok in value[0]: f.write(key+','+tok+'\n')
   
    print(f"number of tokens in val fold {i}: {cnt}")
    
