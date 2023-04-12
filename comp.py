import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os,re
import argparse

def plot_2d_conf_mat(mat):
    tp,fp,fn,tn=mat
    array = [[tp,fp],[fn,tn]]
    index = ["Changed","Unchanged"]
    columns= ["Changed","Unchanged"]
    df_cm = pd.DataFrame(array,index=index,columns=columns)
    # plt.figure(figsize=(10,7))
    #sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()

def get_metrics(err,cln,tar):
    """Take in as input file names of error file, cleaned file and target file. """
    with open(err) as f: 
        X = [line.strip() for line in f]
        #X = " ".join(X).split()
    with open(cln) as f: 
        predict = [line.strip() for line in f]
        #predict = " ".join(predict).split()
    with open(tar) as f: 
        actual = [line.strip() for line in f]
        #actual = " ".join(actual).split()
    #X = x[0]
    #actual = x[1]
    #predict = x[2]
    #X = ["on","tw","thre","four","fice","asd"]
    #actual=["one","two","three","four","five","and"]
    #predict=["on","tw","three","four","fire","and"]
    # cc,ci,ui,uc
    tp,fp,fn,tn=0,0,0,0
    actual_changed,predicted_changed,actual_unchanged,predicted_unchanged,correct,incorrect=0,0,0,0,0,0
    for i in range(len(X)):
        #Correctly changed
        if actual[i] == predict[i]:
            correct+=1
            if predict[i]==X[i]: actual_unchanged+=1
            elif predict[i]!=X[i]: actual_changed+=1
            #if actual[i]==X[i]: actual_unchanged+=1 #already covered
            #elif actual[i]!=X[i]: actual_changed+=1 #already covered
           
        #Incorreclty changed
        if actual[i] != predict[i]:
            incorrect+=1
            if predict[i]==X[i] and actual[i]!=X[i]: predicted_unchanged+=1
            #elif predict[i]==X[i] and actual[i]==X[i]: actual_unchanged+=1 # never happens
            elif predict[i]!=X[i] and actual[i]==X[i]: predicted_changed+=1
            elif predict[i]!=X[i] and actual[i]!=X[i]: actual_changed+=1
    #arr = [[actual_changed,],[],[]]
    tp,fp,fn,tn = actual_changed, predicted_changed, predicted_unchanged, actual_unchanged
    print(f"{tp} {fp}\n{fn} {tn}")
    #plot_2d_conf_mat((tp,fp,fn,tn))
    sensitivity = recall = tp/(tp+fn) # % correctly identifying positives
    specificity = tn/(tn+fp) # % correctly identifying negatives
    precision = tp/(tp+fp) # 
    f1 = 2*(precision*recall)/(precision+recall)
    #print(f"sensitivity: {sensitivity} specificity: {specificity} precision: {precision} f1: {f1}")
    print(f"sensitivity {sensitivity:.4f}, specificity {specificity:.4f}, precision {precision:.4f}, f1 {f1:.4f}")

    TPR = sensitivity
    FPR = 1-specificity #= fp/(fp+tn)
    #print(f"true positive rate: {TPR} false positive rate: {FPR}")
    #x = np.array(X)
    #y_true = np.array(actual)==x
    #y_pred = np.array(predict)==x
    #y_true,y_pred = y_true.astype(int), y_pred.astype(int)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode anahash cleaned files..")
    parser.add_argument("--errfile", help="path to error filled file")
    parser.add_argument("--clnfile", help="path to cleaned file ")
    parser.add_argument("--tarfile", help="path to target file")
    parser.add_argument("--clnfolder",help="path to decoded and cleaned files")
    parser.add_argument("--outfolder",help="path to output folder")

    args = parser.parse_args()
    if args.clnfolder != None:
        dir_list = os.listdir(args.clnfolder)
        dir_list.sort(key=lambda f: int(re.sub('\D', '', f)))
        print("file_name,sensitivity,specificity,precision,f1")
        for fn in dir_list:
            print(fn,end=",")
            get_metrics(err=args.errfile,
                    cln=args.clnfolder+'/'+fn,
                    tar=args.tarfile)
    else: get_metrics(err=args.errfile,cln=args.clnfile,tar=args.tarfile)

