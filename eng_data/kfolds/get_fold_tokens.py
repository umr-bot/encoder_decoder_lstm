"""Extract tokens from fold k of corpora"""
def get_fold_tokens(input_fn="foldset",output_fn="fold_tokens/"):
    kfolds = 5
    for i in range(kfolds):
        with open(input_fn+str(i+1)+"/train") as f:
            tokens = [tok for seq in f for tok in seq.strip('\n').split()]
        with open(output_fn+"fold_tokens_"+str(i+1),"w") as f:
            for tok in tokens: f.write(tok + '\n')
