"""Some scripts to check how many errors are in the folds."""
from balance_errors_in_dataset import Balancer

def print_num_errs(k=6, bal=None):
    """ Print number of errors in folds 1 to k.
        Return: -1 if error
                else print out to std output the number of errors in each fold
    """
    # If no Balancer class is passed to function
    if bal==None: 
        print("ERROR. Must pass a Balancer class as argument to function.")
        return -1

    num_errs=[]
    for i in range(1,k):
        _,_,_,_,err_tups = bal.check_err(input_fn="folds/fold"+str(i),normed_input_fn="norm_folds/fold"+str(i))
        num_errs.append(len(err_tups))
    print(num_errs)

if __name__ == "__main__":
    print_num_errs()
