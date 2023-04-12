# coding: utf-8
from dev_main import Main
from tqdm import tqdm
import os,re

class compare:
    log_dir = "logs_session_Jan_Feb_2021/"
    fns = os.listdir(log_dir)
    ana_main = Main()

    def __init__(self):
        pass

    def get_file_paths(self,path,filetype=".TextGrid"):
        """Get all filenames of type 'filetype' in a directory 
           recursively.
           Take as input the root directory 'path' to search from
           and also the type of files with 'filetype' to look for.
        """
        file_paths = []
        for root, dirs, files in os.walk(path):
            for name in files:
                file_path = os.path.join(root, name)
                if os.path.splitext(file_path)[1] == filetype:
                    file_paths += [file_path]   
        return file_paths

    def get_string_between(self,sent,start,end):
        return (sent.split(start))[1].split(end)[0]

    def get_file_contents_in_dir(self):
        tot_logs = []
        for fn in self.fns:
            with open(self.log_dir+fn) as f:
                tot_logs.append([line for line in f])
        return tot_logs

    def get_glo_log(self,log_file):
        glo = []
        with open(log_file) as f:
            for line in f:
                if ":Globally" in line:
                    source_word = self.get_string_between(line," change "," to ")
                    changed_word = self.get_string_between(line," to "," in ")
                    glo.append((source_word, changed_word))
        return glo
    def ggl(self, line):
        source_word = self.get_string_between(line," change "," to ")
        changed_word = self.get_string_between(line," to "," in ")
        return source_word,changed_word

    def get_penta_log(self, change_log):
        source_word = self.get_string_between(change_log,":Change "," in")
        directory = self.get_string_between(change_log,"file "," interval")
        interval = self.get_string_between(change_log,"interval "," instance")
        instance = self.get_string_between(change_log,"instance "," to")
        changed_word = self.get_string_between(change_log,"to ","\n")

        return source_word,directory,interval,instance,changed_word
    
    def get_context_around_word(self,line,word,instance=1,global_flag=False,ind=0):
        """Depending on length of line, get context around the occurance
           of a specific word."""

        line_toks = line.split()
        if global_flag==False: #local/specific/singular change
            #print(word)
            #print(line)
            ind_list = [index for index, value in enumerate(line_toks) if value == word]
            #print("ind _list",ind_list)
            ind = ind_list[instance-1]
            #print("ind",ind)
        else: pass#ind = ind

        if len(line_toks) >= 5 and ind >= 2 and ind <= len(line_toks)-3:
            return (word,(line_toks[ind-2],line_toks[ind-1],line_toks[ind+1],line_toks[ind+2]))
        elif len(line_toks) >= 3  and ind >= 1 and ind <= (len(line_toks)-2):
            return (word,(line_toks[ind-1],line_toks[ind+1]))
    
    def get_glo_changes(self,glo_word):
        sents_containing_glo_word = []
        fps = self.get_file_paths('textgrids/')
        for fp in fps:
            with open(fp) as f:
                for line in f:
                    if glo_word in line:
                        sents_containing_glo_word.append(self.get_string_between(line,"= \"","\" \n")
)
        return sents_containing_glo_word
    
    def get_change_logs(self,logs):
        """Get all lines in logs which either a global or local change is made"""
        logs_5 = [] # source_word,directory,interval,instance,changed_word
        context = []
        for i in range(len(logs)):
            #print(logs[i])
            #break
            if ":Change" in logs[i]:
                penta_log = source_word,directory,interval,instance,changed_word= self.get_penta_log(logs[i])
                logs_5.append(penta_log)
                with open(directory) as f:
                    #textgrid = [line.replace('"', '') for line in f]
                    textgrid = [line for line in f]

                line = ""
                for tg_line in range(len(textgrid)):
                    if "intervals ["+interval+']' in textgrid[tg_line]: 
                        #line = textgrid[tg_line+3] # index 0 and 1 contains strings "text" "="
                        line = self.get_string_between(textgrid[tg_line+3],'= "','\"') # Plus 3 is to move pointer to the line containing text. Check textgrid format.
                        if line == ['JUNK']: continue
                        #print("local line",line)
                        context.append([self.get_context_around_word(line=line,word=source_word,instance=int(instance))])

            if ":Globally" in logs[i]:
                glo_source_word, glo_changed_word = self.ggl(logs[i])
                logs_5.append((glo_source_word,glo_changed_word))
                sents_containing_source_word = self.get_glo_changes(glo_word=glo_source_word)
                #print(sents_containing_source_word,glo_source_word)
                #print("glo_word",glo_source_word)
                glo_contexts = []
                for sent in sents_containing_source_word:
                    #print("sent",sent)
                    ind_list = [index for index, value in enumerate(sent.split()) if value == glo_source_word]
                    glo_context = None
                    for ind in ind_list: glo_context = self.get_context_around_word(sent,glo_source_word,global_flag=True,ind=ind)
                    #print("glo_context",glo_context)
                    if glo_context != None: glo_contexts.append(glo_context)
                context.append(glo_contexts)
        return logs_5,context

    def get_context(self):
        logs_5 = [] # source_word,directory,interval,instance,changed_word
        context = []
        tot_logs = self.get_file_contents_in_dir()
        for log_ind in range(len(tot_logs)):
            logs_5_, context_ = self.get_change_logs(tot_logs[log_ind])
            logs_5.append(logs_5_)
            context.append(context_)
        logs_5[:] = [item for item in logs_5 if (item!=[] and item != None)]
        #logs_5 = [val for sublist in logs_5 for val in sublist]
        context[:] = [item for item in context if (item != [] and item != None)]

        return logs_5,context

    def get_ana_word(self,context,context_flag=True):
        """Compare anahash correction against corrections made by human linguist
           Return 1 if same answer is reached and 0 if different results.
        """
        #fw = logs_5[log_num][-1] #TODO: should it be 0 in place of -1 ?
        if context_flag == True:
            fw = context[0]
            lv = self.ana_main.get_likely_variants(fw_list=[fw])

            #d = self.ana_main.upgrade(fw=fw,lv=lv,w=(context[1]),zipf=self.ana_main.get_zipf(len(fw)), context=True)
            #b1 = self.ana_main.get_best(fw=fw,d=d,lpc_rpc_flag=False)
            ana_word,d = self.ana_main.best_match(fw=fw,with_context=False)
        else:
            fw = context
            lv = self.ana_main.get_likely_variants(fw_list=[fw])
            #d = self.ana_main.upgrade(fw=fw,lv=lv,zipf=self.ana_main.get_zipf(len(fw)), context=False)
            #b1 = self.ana_main.get_best(fw=fw,d=d,lpc_rpc_flag=False)
            ana_word,d = self.ana_main.best_match(fw=fw,with_context=False)

        #print(bm)
        #print(logs_5[log_num][-1])
        #if ana_word == logs_5[log_num][-1]: return 1,ana_word
        #else: return 0,ana_word
        return ana_word
    def get_compare_vals(self):
        logs_5,contexts = self.get_context()
        comp_vals = []
        print("lenlog",len(logs_5))
        for i in tqdm(range(len(logs_5)),colour="red"):
            if logs_5[i] == [] or contexts[i] == []: continue
            # note source_word = logs_5[i][0] = context[i][j][0]
            # for one source word there exists many context tuples
            for j in tqdm(range(len(contexts[i])),colour="blue"):
                source_word, changed_word = logs_5[i][j][0],logs_5[i][j][-1]
                for k in tqdm(range(len(contexts[i][j])),colour="green"):
                    if contexts[i][j][k] == None or contexts[i][j][k] == []: continue
                    ana_word = self.get_ana_word(context=contexts[i][j][k])
                    comp_vals.append((source_word,changed_word,ana_word))
            
        return comp_vals
#c = compare()
#comp_vals = c.get_compare_vals()
#with open("temp5_comp_vals.py",'w') as f:
#    for tup in comp_vals:
#        f.write('"'+tup[0]+'"'+','+'"'+tup[1]+'"'+','+'"'+tup[2]+'"'+'\n')

#os.system("shutdown /s /t 1")
