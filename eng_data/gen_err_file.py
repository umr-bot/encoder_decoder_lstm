# coding: utf-8
from add_spelling_error import add_spelling_erors
import argparse

def gen_error_file(infile,outfile,error_rate=0.8):
    """Input: name of a file that errors are to be added to
      Output: name of file to save to"""
    with open(infile) as f:
        in_data = [line.strip() for line in f]

    #chars = list(set(' '.join(in_data)))
    chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ\'" )
    err_lines = []
    for line in in_data:
        err_line = []
        for tok in line.split():
            err_tok = add_spelling_erors(tok,error_rate,chars)
            err_line.append(err_tok)
        err_lines.append(err_line)
        
    with open(outfile,'w') as f:
        for line in err_lines:
            for tok in line:
                f.write(tok + " ")
            f.write('\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate error version of an input file.")
    parser.add_argument("--infile", required=True, help="name of input file")
    parser.add_argument("--outfile", required=True, help="name of output file")
    parser.add_argument("--error_rate",help="relative rate of errrors added")
    args = parser.parse_args()
    if args.error_rate != None: gen_error_file(args.infile,args.outfile,error_rate=float(args.error_rate))
    else: gen_error_file(args.infile,args.outfile)

