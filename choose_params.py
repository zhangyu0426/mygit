##first "ls ./benchmark/>filelist.txt"

import os
import sys
import numpy as np

os.system('ls ./benchmark/ > filelist.txt')
name_all=[]
val_all=[]
bm_dir="./benchmark/"
files=open('filelist.txt','r')
for line in files:
    line=line.strip()
    name=line.split('.t')[0]
    print name
    name_all.append(name)
    
    path_onefile=bm_dir+line
    onefile=open(path_onefile,'r')
    for line2 in onefile:
        line2=line2.strip()
        info=line2.split('\t')
        val=info[1]
        print val
        val_all.append(val)
    onefile.close()
files.close()

print len(name_all)
print len(val_all)
    
sorted_list=sorted(range(len(val_all)),key=lambda k:val_all[k],reverse=True)
print sorted_list

topk=15
for i in range(topk):
    print ("%s : %s\n" %(name_all[sorted_list[i]],val_all[sorted_list[i]]))
