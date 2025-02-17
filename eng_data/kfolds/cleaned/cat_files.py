"""Concatenate cleaned batch files that are generated by clean.py function in
   root/src dir of project.
"""
import os,re,subprocess
from tqdm import tqdm
#import readline # to enable autocomplete for raw_input
#readline.parse_and_bind("tab: complete")

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

fold_dirs = [fold_dir for fold_dir in os.listdir() if os.path.isdir(fold_dir)]
fold_dirs = natural_sort(fold_dirs)
# [a for a in os.listdir(currentdir) if os.path.isdir(os.path.join(currentdir, a))] # USEFUL ADDITIONAL INFO if want to get sub_dir also

for fold_dir in tqdm(fold_dirs):
    batch_dir = fold_dir + "/batch_files/"
    fol_names = natural_sort(os.listdir(batch_dir))

    for fol_name in tqdm(fol_names):
        batch_fns = natural_sort(os.listdir(batch_dir + fol_name)) # file names in directory fol_name
        for batch_fn in batch_fns:
            with open(batch_dir+fol_name+'/'+batch_fn, "rb", 0) as a, open(fold_dir+"/cat_files/" + fol_name+".txt", "a") as b:
                rc = subprocess.call(["cat"], stdin=a, stdout=b)

