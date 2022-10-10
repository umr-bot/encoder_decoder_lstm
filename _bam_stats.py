# coding: utf-8
from sort_error_types import DataHandler, distribute, split
from collections import defaultdict
from tqdm import tqdm

data_handler = DataHandler()
sorted_single_errs, err_counts = data_handler.get_err_counts_dicts()
fold_lens = sum(range(10)) + sum(range(11,19)) +21+23+26+29, sum([31,32,33,36,44,48,60]), sum([73,86,92]), 503
num_folds = 4
fold1 = [*range(1,5)]#+[*range(11,19)]#+[21,23,26,29]
fold1_dev=[5]
fold1_test=[6]
fold2 = [*range(7,11)]
fold2_dev=[11]
fold2_test=[12,13]
fold3 = [21,23,26,29,31,32,33,36]
fold3_dev=[44]
fold3_test=[48,60,73]
fold4 = [*range(14,15)]+[503]
fold4_dev=[15,86]
fold4_test=[16,17,18,92]

fold_lst = [fold1,fold2,fold3,fold4]
fold_devs = [fold1_dev,fold2_dev,fold3_dev,fold4_dev]
fold_tests = [fold1_test,fold2_test,fold3_test,fold4_test]
folds = defaultdict(list)
for i in range(num_folds):
    folds["fold"+str(i+1)].append(fold_lst[i])
    folds["fold"+str(i+1)+"dev"].append(fold_devs[i])
    folds["fold"+str(i+1)+"test"].append(fold_tests[i])

err_folds=defaultdict(list)
#err_folds=[[] for dummy in range(len(folds))]
for single_err in sorted_single_errs:
    err_tok = single_err[0][2][1]
    err_count = err_counts[err_tok]
    for folds_key in folds.keys():
        if err_count in folds[folds_key][0]:
            err_folds[folds_key].append(single_err)
            #err_folds[fold_cnt].append(single_err)

print("number of sentences with unequal sequence lengths vs total fold size per fold:")
for fold_key in folds.keys():
    err_fold = err_folds[fold_key]
    y = [x for x in err_fold if len(x[0][0])!=len(x[0][1])]
    print(f"{fold_key}: {len(y)}, {len(err_fold)}")

# Sanity check to see that there is no overlapping of types between folds
# Note however that lists in x below are token lists and not type lists
x = [[] for dummy in range(num_folds)]
for i in range(num_folds):
    x[i] = [err_fold[0][2][1] for err_fold in err_folds["fold"+str(i+1)]]
def rotate(l, n):
    return l[n:] + l[:n]
def flatten(l):
    return [item for sublist in l for item in sublist]
# All these statements should print True if all folds disjoint
if set(tuple(flatten(x[1:4]))).isdisjoint(x[0]): print("fold1 disjoint of all other folds")
if set(tuple(flatten(rotate(x,1)[1:4]))).isdisjoint(x[1]): print("fold2 disjoint of all other folds")

if set(tuple(flatten(rotate(x,2)[1:4]))).isdisjoint(x[2]): print("fold3 disjoint of all other folds")

if set(tuple(flatten(rotate(x,3)[1:4]))).isdisjoint(x[3]): print("fold4 disjoint of all other folds")

def write_folds(err_folds):
    for fold_key in err_folds.keys():
        i = str(int(fold_key[4]))
        fn = "bam_folds/"+"fold"+str(i)+"/"+fold_key
        with open(fn,"w") as f:
            trains = [x[0][0] for x in err_folds[fold_key]]
            for train in trains: f.write(train+"\n")
        norm_fn = "bam_folds/"+"norm_fold"+str(i)+"/"+fold_key
        with open(norm_fn,"w") as f:
            norm_trains = [x[0][1] for x in err_folds[fold_key]]
            for norm_train in norm_trains: f.write(norm_train+"\n")

def join_data(err_folds,without_err,norm_without_err):
    err_folds, counts = zip(*err_folds)
    errors,norm_errors,error_tups_reformed = zip(*err_folds)
    assert(len(without_err)==len(norm_without_err))

    sorted_list = []
    if len(without_err) > len(err_folds): 
        sorted_list = list(distribute(source_one=without_err,source_two=errors))
        norm_sorted_list = list(distribute(source_one=norm_without_err,source_two=norm_errors))
        print("Less sentences with errors than those without.")
    elif len(without_err) == len(err_folds):
        # Add empty sentences to source_one arguments for distribute function to
        # work, source_one > source_two, hence we add empty sentence
        sorted_list = distribute(source_one=without_err.append(''),source_two=errors)
        norm_sorted_list = distribute(source_one=norm_without_err.append(''),source_two=norm_errors)
        print("Sentences with errors the same amount as those without.")
    else: # reverse order of arguments passed to distribute function
          # as len(source_one) > len(source_two) for function to work
        sorted_list = distribute(source_one=errors,source_two=without_err)
        norm_sorted_list = distribute(source_one=norm_errors,source_two=norm_without_err)
        print("More sentences with errors than those without.")

    return sorted_list, norm_sorted_list

ll = list(set([tok for toks in data_handler.without_err for tok in toks.split()]))
interval = int(len(ll)/num_folds)
ll_spliced=[]
e = defaultdict(set)
for i in range(num_folds):
    if i == num_folds-1: ll_spliced = ll[i*interval:]
    else: ll_spliced = ll[i*interval:(i+1)*interval]
    # Add 250 tuples from correct tokens to dev and test
    dev = list(tok[0][2][1] for tok in err_folds["fold"+str(i+1)+"dev"])
    test = list(tok[0][2][1] for tok in err_folds["fold"+str(i+1)+"test"])
    dev_test = dev + test
    e["fold"+str(i+1)+"dev"] = set([tup[0][2] for tup in err_folds["fold"+str(i+1)+"dev"]] + [(x,x) for x in ll_spliced[-1000:-500]])
    e["fold"+str(i+1)+"test"] = set([tup[0][2] for tup in err_folds["fold"+str(i+1)+"test"]] + [(x,x) for x in ll_spliced[-500:]])
    e["fold"+str(i+1)] = set([tup[0][2] for tup in err_folds["fold"+str(i+1)]])
    # note union does not do in place updates, hence have to assign result
    e["fold"+str(i+1)]=e["fold"+str(i+1)].union([(x,x) for x in ll_spliced[0:-1000] if x not in dev_test])
#    e[key_name].union(set(tup[0][2] for tup in err_folds[key_name] if tup[0][2] not in ll))

#ll=list(ll)
#interval = int(len(ll)/num_folds)
#for i in range(num_folds):
#    if i == (num_folds-1): e["fold"+str(i)].update(ll[interval:])
#    e["fold"+str(i)].union(ll[interval:i*interval])

#e = defaultdict(list)
#for key_name in err_folds.keys():
#    err_ = set(x[0][2][0] for x in err_folds[key_name])
#    norm_ = set(x[0][2][1] for x in err_folds[key_name])
#    e[key_name] = [err_,norm_]
#    ll_and = ll.intersection(norm_)
#    if len(ll_and) > 0:
#        print(f"making {key_name} disjoint of without error data")
#        for el in ll_and: 
#            for i in range(len(err_folds[key_name])):
#                err_fold0 = err_folds[key_name][i][0][2][0]
#                err_fold1 = err_folds[key_name][i][0][2][1]
#                if err_fold != el:
#                    e[key_name].append((err_fold0, err_fold1))
                    #err_folds[key_name].pop(i)
                    #i -= 1
            #e[key_name].remove(el)

#e = defaultdict(set)
#for i in range(1,num_folds+1):
#    fold_name = "fold"+str(i)
#    e1 = set([x[0][2][1] for x in err_folds[fold_name]])
#    e2 =set([x[0][2][1] for x in err_folds[fold_name+"dev"]])
#    e3 =set([x[0][2][1] for x in err_folds[fold_name+"test"]])
#    err_fold = e1.union(e2).union(e3)
#    e[fold_name].update(err_fold)

#without_err_folds = split(without_err,k_folds=4)
#norm_without_err_folds = split(norm_without_err,k_folds=4)

#sorted_folds,norm_sorted_folds=[],[]
#for     
#    sorted_list,norm_sorted_list = join_data(err_folds["fold1"], without_err_folds[0], norm_without_err_folds[0])


