# tups[][0] == error and tups[][1] ==norm
with open("all") as f: tups = [line.strip('\n') for line in f]

num_folds = 5
parts = [[] for i in range(num_folds)]
div = len(tups)//num_folds

parts[0] = tups[0:div]      
parts[1] = tups[div:div*2]  
parts[2] = tups[div*2:div*3]
parts[3] = tups[div*3:div*4]
parts[4] = tups[div*4:]       
for i in range(num_folds):
    with open("folds/part"+str(i+1),'w') as f:        
        for line in parts[i]: f.write(line + '\n')

